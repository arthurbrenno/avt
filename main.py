import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
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
        'names': {i: class_name for i, class_name in enumerate(classes)}
    }

    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)

def main():
    # Defina o diretório base dos dados de forma relativa ao script
    base_data_dir = Path(__file__).parent / "data/archive"

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
    # Use um modelo padrão ou forneça o caminho correto para o seu modelo
    model = YOLO('models/yolov10l.pt')  # Substitua pelo caminho correto se necessário

    # Iniciar o treinamento
    model.train(
        data='data.yaml',
        epochs=100,               # Ajuste conforme necessário
        imgsz=640,                # Tamanho da imagem
        batch=16,                 # Tamanho do batch
        name='yolov10l_trained',   # Nome do experimento
        project='runs/train',     # Diretório de saída
        device='cuda',             # GPU a ser usada, 'cpu' para CPU
        verbose=True
    )

if __name__ == "__main__":
    main()
