# 01_llenar_tablas_with_gan.py
import pandas as pd
import numpy as np
import os
from faker import Faker
import random
from datetime import datetime, timedelta
from pathlib import Path

# GAN (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------
# Configuración / semillas
# ------------------------
faker_gen = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# ------------------------
# Parámetros generales
# ------------------------
OUTDIR = Path(".")
script_dir = Path(__file__).parent
n_total = 200_000            # reduce si tu máquina es modesta; puedes aumentar luego
frac_fraud_initial = 0.10    # proporción inicial de fraudes creados por reglas
n_fraud = int(n_total * frac_fraud_initial)
n_no_fraud = n_total - n_fraud

tipo_tarjeta_choices = ['clasica', 'oro', 'platinum']
estado_tarjeta_choices = ['activa', 'bloqueada', 'vencida']
tipo_nomina_choices = ['gobierno', 'privada', 'independiente']
ind_estado_choices = ['al dia', 'en mora', 'castigada']

# ------------------------
# Funciones auxiliares (mantengo tus reglas)
# ------------------------
def generar_fecha_vencimiento():
    hoy = datetime.today()
    return hoy + timedelta(days=random.randint(-365, 1095))

def generar_registro(fraudulento):
    while True:
        estado_tarjeta = random.choices(estado_tarjeta_choices, weights=[0.7, 0.2, 0.1])[0]
        acumulado_cupo = round(np.random.beta(2, 5), 2)
        cuotas_mora = np.random.poisson(1)
        es_reexpedicion = random.choices([0, 1], weights=[0.85, 0.15])[0]
        es_amparada = random.choices([0, 1], weights=[0.8, 0.2])[0]
        indicador_repositorio = random.choices([0, 1], weights=[0.9, 0.1])[0]
        tipo_nomina = random.choice(tipo_nomina_choices)
        localizacion_tarjeta = random.choices(['nacional', 'internacional'], weights=[0.85, 0.15])[0]

        if estado_tarjeta in ['bloqueada', 'vencida'] and random.random() < 0.6:
            fecha_ult_retiro = None
        else:
            fecha_ult_retiro = faker_gen.date_between(start_date='-180d', end_date='today')

        registro = {
            'Numero_tarjeta': faker_gen.sha256()[0:12],
            'Tipo_tarjeta': random.choice(tipo_tarjeta_choices),
            'Estado_tarjeta': estado_tarjeta,
            'Fecha_vencimiento': generar_fecha_vencimiento().date(),
            'Indicador_repositorio': indicador_repositorio,
            'Localizacion_tarjeta': localizacion_tarjeta,
            'Acumulado_cupo': acumulado_cupo,
            'Fecha_ult_retiro': fecha_ult_retiro,
            'Cuotas_mora': int(cuotas_mora),
            'Ind_estado': random.choice(ind_estado_choices),
            'Es_amparada': es_amparada,
            'Es_reexpedicion': es_reexpedicion,
            'Tipo_nomina': tipo_nomina
        }

        es_fraude = int(
            (estado_tarjeta == 'bloqueada' and es_reexpedicion == 1) or
            (cuotas_mora >= 3 and acumulado_cupo > 0.9) or
            (indicador_repositorio == 1 and localizacion_tarjeta == 'internacional') or
            (es_amparada == 1 and es_reexpedicion == 1) or
            (fecha_ult_retiro is None and estado_tarjeta != 'activa') or
            (tipo_nomina == 'independiente' and cuotas_mora > 4)
        )

        if (fraudulento and es_fraude == 1) or (not fraudulento and es_fraude == 0):
            registro['fraude'] = es_fraude
            return registro

# ------------------------
# Generación masiva por reglas (como ya hacías)
# ------------------------
print("Generando dataset base (reglas + faker)...")
fraudes = [generar_registro(fraudulento=True) for _ in range(n_fraud)]
no_fraudes = [generar_registro(fraudulento=False) for _ in range(n_no_fraud)]
df = pd.DataFrame(fraudes + no_fraudes).sample(frac=1, random_state=42).reset_index(drop=True)

# guardar versión sin ruido
out_clean = script_dir / "tarjetas_fraude_base.csv"
df.to_csv(out_clean, index=False)
print("Guardado base:", out_clean)

# ------------------------
# Introducir ruido de etiquetas (20%)
# ------------------------
noise_rate = 0.20
df['fraude_true'] = df['fraude']
n_total_actual = len(df)
total_noise = int(noise_rate * n_total_actual)
fp_idx = df.loc[df['fraude_true'] == 0].sample(n=total_noise//2, random_state=42).index
fn_idx = df.loc[df['fraude_true'] == 1].sample(n=total_noise - len(fp_idx), random_state=42).index
df.loc[fp_idx, 'fraude'] = 1
df.loc[fn_idx, 'fraude'] = 0
df['es_etiqueta_ruidosa'] = (df['fraude'] != df['fraude_true']).astype(int)

# ------------------------
# Preparar datos numéricos para entrenar la GAN (codificar y escalar)
# ------------------------
print("Codificando y escalando features para entrenar GAN...")
X = df.drop(columns=['Numero_tarjeta', 'fraude_true', 'es_etiqueta_ruidosa', 'fraude'])
X = pd.get_dummies(X, drop_first=True)
# Guardar encoder columns para usar luego en detector
cols = X.columns.tolist()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# dataset de fraudes reales (post-ruido: usamos etiqueta 'fraude_true' para entrenar GAN con ground truth real)
fraud_mask = df['fraude_true'] == 1
X_fraud = X_scaled[fraud_mask.values]
print("Ejemplos fraude reales para GAN:", X_fraud.shape[0])

# ------------------------
# Definir una GAN simple (MLP) para features tabulares
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 32
feat_dim = X_scaled.shape[1]

class Generator(nn.Module):
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

G = Generator(z_dim, feat_dim).to(device)
D = Discriminator(feat_dim).to(device)
optimG = torch.optim.Adam(G.parameters(), lr=1e-4)
optimD = torch.optim.Adam(D.parameters(), lr=1e-4)
bce = nn.BCELoss()

# DataLoader fraudes
if X_fraud.shape[0] < 10:
    print("ADVERTENCIA: pocos fraudes reales; la GAN puede no generalizar bien.")
fraud_loader = DataLoader(TensorDataset(torch.tensor(X_fraud, dtype=torch.float32)),
                          batch_size=64, shuffle=True, drop_last=True)

# Entrenamiento GAN (pocas épocas por defecto; aumenta si GPU)
n_epochs = 1
print("Entrenando GAN sobre fraudes reales (puede tardar)...")
for epoch in range(n_epochs):
    for (real_batch,) in fraud_loader:
        real = real_batch.to(device)
        b = real.size(0)
        # Train D
        D.zero_grad()
        label_real = torch.ones((b,1), device=device)
        out_real = D(real)
        lossD_real = bce(out_real, label_real)
        z = torch.randn(b, z_dim, device=device)
        fake_tensor = G(z).detach()  # evitar usar nombre 'fake' para no confundir con faker_gen
        label_fake = torch.zeros((b,1), device=device)
        out_fake = D(fake_tensor)
        lossD_fake = bce(out_fake, label_fake)
        lossD = (lossD_real + lossD_fake) * 0.5
        lossD.backward()
        optimD.step()
        # Train G
        G.zero_grad()
        z = torch.randn(b, z_dim, device=device)
        gen = G(z)
        label_gen = torch.ones((b,1), device=device)
        out_gen = D(gen)
        lossG = bce(out_gen, label_gen)
        lossG.backward()
        optimG.step()
    if (epoch+1) % 100 == 0 or epoch==0:
        print(f"Epoch {epoch+1}/{n_epochs}  lossD={lossD.item():.4f} lossG={lossG.item():.4f}")

# ------------------------
# Generar muestras sintéticas de fraude y desescalar + reconstruir columnas
# ------------------------
n_generate = int(0.5 * (len(df) - X_fraud.shape[0]))  # por ejemplo generar hasta 50% del resto
print("Generando muestras sintéticas de fraude:", n_generate)
G.eval()
with torch.no_grad():
    z = torch.randn(n_generate, z_dim, device=device)
    gen_feats = G(z).cpu().numpy()

# desescalar gen_feats (scaler fue fit sobre X)
gen_feats_unscaled = scaler.inverse_transform(gen_feats)
df_gen = pd.DataFrame(gen_feats_unscaled, columns=cols)

# reconstruir columnas categóricas: para columnas dummy, tomar la máxima probabilidad por grupo
# (esto es un atajo: convertimos numéricos a categorías por reglas simples)
# Para seguridad, vamos a clippear valores y luego intentar mapear:
for c in df_gen.select_dtypes(include=[np.number]).columns:
    if '_ ' in c or c.startswith('Tipo_tarjeta') or c.startswith('Estado_tarjeta') or c.startswith('Tipo_nomina') or c.startswith('Localizacion_tarjeta') or c.startswith('Ind_estado'):
        # no hacemos nada sofisticado aquí; mantenemos la columna numérica: el detector puede usar dummies
        pass
# Añadir columnas obligatorias que tenías (Numero_tarjeta, Fecha_vencimiento, Fecha_ult_retiro)
df_gen['Numero_tarjeta'] = [faker_gen.sha256()[0:12] for _ in range(len(df_gen))]
df_gen['Fecha_vencimiento'] = [(datetime.today() + timedelta(days=random.randint(-365, 1095))).date() for _ in range(len(df_gen))]
df_gen['Fecha_ult_retiro'] = [faker_gen.date_between(start_date='-180d', end_date='today') for _ in range(len(df_gen))]
# etiqueta de fraude sintético
df_gen['fraude'] = 1
df_gen['fraude_true'] = 1
df_gen['es_etiqueta_ruidosa'] = 0

# Para concatenar, aseguramos que columnas se alineen (llenos NaN con 0 / apropiado)
# Asegurar que df_gen tenga todas las columnas categóricas originales
df_gen['Tipo_tarjeta'] = [random.choice(tipo_tarjeta_choices) for _ in range(len(df_gen))]
df_gen['Estado_tarjeta'] = [random.choice(estado_tarjeta_choices) for _ in range(len(df_gen))]
df_gen['Localizacion_tarjeta'] = [random.choice(['nacional','internacional']) for _ in range(len(df_gen))]
df_gen['Ind_estado'] = [random.choice(ind_estado_choices) for _ in range(len(df_gen))]
df_gen['Tipo_nomina'] = [random.choice(tipo_nomina_choices) for _ in range(len(df_gen))]

# Reordenar columnas para que coincidan exactamente con df
df_gen = df_gen.reindex(columns=df.columns)

# Concatenar
df_final = pd.concat([df, df_gen], ignore_index=True, sort=False)\
             .sample(frac=1, random_state=42).reset_index(drop=True)

out_noisy_aug = script_dir / "tarjetas_fraude_con_ruido_20pct_augmented.csv"
df_final.to_csv(out_noisy_aug, index=False)
print("✅ Dataset final con muestras sintéticas guardado en:", out_noisy_aug)
