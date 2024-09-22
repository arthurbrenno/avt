import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import argparse
from ultralytics import YOLO

def parse_xml_annotations(data_dir, classes_set):
    """
    Parse XML files and extract class names.

    Args:
        data_dir (Path): Path to the dataset directory (train, val, test).
        classes_set (set): Set to store unique class names.

    Returns:
        list: List of image file paths.
    """
    image_paths = []
    xml_files = list(data_dir.rglob("*.xml"))  # Busca recursiva

    print(f"Processando {len(xml_files)} arquivos XML em {data_dir}")

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                name_tag = obj.find('name')
                if name_tag is not None and name_tag.text:
                    class_name = name_tag.text.strip().lower()
                    classes_set.add(class_name)
                    print(f"Encontrada classe: {class_name} no arquivo {xml_file.relative_to(data_dir)}")
                else:
                    print(f"Tag <name> não encontrada em {xml_file.relative_to(data_dir)}")

            # Assuming images have the same name as XML but with .jpg extension
            image_file = xml_file.parent / (xml_file.stem + ".jpg")
            if image_file.exists():
                image_paths.append(str(image_file))
            else:
                print(f"Imagem correspondente não encontrada para {xml_file.relative_to(data_dir)}")
        except ET.ParseError as e:
            print(f"Erro ao analisar {xml_file.relative_to(data_dir)}: {e}")

    return image_paths

def convert_xml_to_yolo(data_dir, classes_dict):
    """
    Convert XML annotations to YOLO format (.txt).

    Args:
        data_dir (Path): Path to the dataset directory (train, val, test).
        classes_dict (dict): Dictionary mapping class names to IDs.
    """
    xml_files = list(data_dir.rglob("*.xml"))

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)

            yolo_annotations = []

            for obj in root.findall('object'):
                name_tag = obj.find('name')
                if name_tag is not None and name_tag.text:
                    class_name = name_tag.text.strip().lower()
                    if class_name not in classes_dict:
                        print(f"Classe '{class_name}' não encontrada no dicionário de classes.")
                        continue
                    class_id = classes_dict[class_name]
                else:
                    print(f"Tag <name> não encontrada em {xml_file.relative_to(data_dir)}")
                    continue

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

            # Escrever as anotações no arquivo .txt correspondente
            txt_file = xml_file.parent / (xml_file.stem + ".txt")
            with open(txt_file, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + "\n")

        except ET.ParseError as e:
            print(f"Erro ao analisar {xml_file.relative_to(data_dir)}: {e}")

def create_data_yaml(base_dir, classes):
    """
    Create the data.yaml file for YOLO training.

    Args:
        base_dir (Path): Base directory containing train, val, test folders.
        classes (list): List of class names.
    """
    data = {
        'path': str(base_dir.resolve()),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': classes  # Alterado para uma lista, conforme esperado pelo YOLO
    }

    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)

def main():
    parser = argparse.ArgumentParser(
        description="Script para converter anotações XML para YOLO e treinar um modelo YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Caminho para o arquivo do modelo YOLO (.pt) a ser treinado.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/archive',
        help='Caminho para o diretório base dos dados.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número de épocas para treinamento.'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Tamanho da imagem para treinamento.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Tamanho do batch para treinamento.'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='yolov_trained',
        help='Nome do experimento de treinamento.'
    )
    parser.add_argument(
        '--project-dir',
        type=str,
        default='runs/train',
        help='Diretório de saída para os resultados do treinamento.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Dispositivo a ser usado para treinamento ('cuda' ou 'cpu')."
    )

    args = parser.parse_args()

    # Defina o diretório base dos dados
    base_data_dir = Path(args.data_dir)

    # Verifique se o diretório existe
    if not base_data_dir.exists():
        print(f"Diretório base {base_data_dir} não encontrado.")
        return

    # Defina as subpastas
    subsets = ['train', 'val', 'test']

    # Coletar todas as classes únicas
    classes_set = set()

    # Coletar todas as imagens e extrair classes
    for subset in subsets:
        subset_dir = base_data_dir / subset
        if not subset_dir.exists():
            print(f"Subdiretório {subset_dir} não encontrado. Pulando...")
            continue
        parse_xml_annotations(subset_dir, classes_set)

    # Ordenar as classes para consistência
    classes = sorted(list(classes_set))
    classes_dict = {class_name: idx for idx, class_name in enumerate(classes)}

    print(f"Classes encontradas: {classes_dict}")

    if not classes_dict:
        print("Nenhuma classe encontrada. Verifique os arquivos XML.")
        return

    # Converter todas as anotações XML para YOLO .txt
    for subset in subsets:
        subset_dir = base_data_dir / subset
        if not subset_dir.exists():
            continue
        convert_xml_to_yolo(subset_dir, classes_dict)

    # Criar o arquivo data.yaml
    create_data_yaml(base_data_dir, classes)

    print("Arquivo data.yaml criado com sucesso.")

    # Iniciar o treinamento
    model_path = args.model_path
    model = YOLO(model_path)  # Usa o caminho fornecido via CLI

    # Iniciar o treinamento
    model.train(
        data='data.yaml',
        epochs=args.epochs,               # Ajustado via argumento
        imgsz=args.imgsz,                 # Ajustado via argumento
        batch=args.batch_size,            # Ajustado via argumento
        name=args.output_name,            # Ajustado via argumento
        project=args.project_dir,         # Ajustado via argumento
        device=args.device,               # Ajustado via argumento
        verbose=True
    )

if __name__ == "__main__":
    main()
