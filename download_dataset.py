import os
import zipfile
import shutil
import kagglehub

def baixar_e_extrair_dataset():
    print("📥 Baixando o dataset do Kaggle...")
    dataset_path = kagglehub.dataset_download("andrewmvd/leukemia-classification")
    print(f"📦 Arquivos baixados para: {dataset_path}")

    # Cria pasta de destino para extração
    extract_path = os.path.join("data", "raw")
    os.makedirs(extract_path, exist_ok=True)

    # Verifica arquivos .zip na pasta baixada
    zip_files = [f for f in os.listdir(dataset_path) if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError("❌ Nenhum arquivo .zip encontrado no dataset baixado.")

    # Extrai o .zip
    zip_path = os.path.join(dataset_path, zip_files[0])
    print(f"🔓 Extraindo {zip_path} para {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Normaliza nomes e estrutura: move subpastas para raw/
    root_folders = [f for f in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, f))]
    if len(root_folders) == 1 and "train" not in root_folders[0].lower():
        full_path = os.path.join(extract_path, root_folders[0])
        for item in os.listdir(full_path):
            shutil.move(os.path.join(full_path, item), extract_path)
        shutil.rmtree(full_path)

    # Verifica se pastas 'train' e 'test' existem
    if not os.path.exists(os.path.join(extract_path, "train")):
        raise FileNotFoundError("❌ Pasta 'train' não encontrada após extração.")
    if not os.path.exists(os.path.join(extract_path, "test")):
        raise FileNotFoundError("❌ Pasta 'test' não encontrada após extração.")

    print("✅ Dataset extraído e organizado em: data/raw")

if __name__ == "__main__":
    baixar_e_extrair_dataset()
