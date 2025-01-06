import json
import re
import os
import argparse

parser = argparse.ArgumentParser(description="Process JSON files and generate results.")
parser.add_argument("--questions", type=str, required=True, help="Path to the questions.jsonl file")
parser.add_argument("--meta", type=str, required=True, help="Path to the vip-bench-meta-data.json file")
parser.add_argument("--source", type=str, required=True, help="Path to the source_image folder")
parser.add_argument("--output", type=str, required=True, help="Path to save the output results.jsonl file")

args = parser.parse_args()

a_json_path = args.questions
vip_bench_meta_data_path = args.meta
source_image_folder = args.source
output_jsonl_path = args.output

a_json_data = []
with open(a_json_path, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        a_json_data.append(data)


with open(vip_bench_meta_data_path, 'r') as file:
    vip_bench_meta_data = json.load(file)

results = []

# Process each image in the questions.jsonl file
for image_data in a_json_data:
    image_name = image_data['image']
    question_id = image_data['question_id']
    category = image_data['category']

    # Find the corresponding entry in vip-bench-meta-data.json
    matched_data = None
    for key, value in vip_bench_meta_data.items():
        if value['image'] == image_name:
            matched_data = value
            break

    if not matched_data:
        print(f"Could not find a matching image: {image_name}")
        continue

    # Extract <obj> from the question text
    question = matched_data['question']
    objs = re.findall(r'<(obj\d*)>', question)

    # Locate the corresponding source image JSON file
    image_source = matched_data['image_source']
    source_image_path = os.path.join(
        source_image_folder,
        image_source.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json')
    )

    if not os.path.exists(source_image_path):
        print(f"Source image JSON file not found: {source_image_path}")
        continue

    # Load the source image JSON file
    with open(source_image_path, 'r') as file:
        source_image_data = json.load(file)

    # Find bounding boxes for each <obj>
    obj_boxes = {}
    for obj in objs:
        bbox = None
        for shape in source_image_data['shapes']:
            if shape['label'] == obj:
                points = shape['points']
                shape_type = shape['shape_type']
                if shape_type == 'rectangle':
                    x_min = min(points[0][0], points[1][0])
                    y_min = min(points[0][1], points[1][1])
                    x_max = max(points[0][0], points[1][0])
                    y_max = max(points[0][1], points[1][1])
                    bbox = [x_min, y_min, x_max, y_max]
                elif shape_type == "polygon":
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                    bbox = [x_min, y_min, x_max, y_max]
                else:
                    print(f"Unknown shape_type: {shape_type}")
                break

        if bbox:
            obj_boxes[obj] = bbox

    # Append the result for this image
    results.append({
        'image': source_image_data['imagePath'],
        'text': question,
        'question_id': question_id,
        'category': category,
        'obj_boxes': obj_boxes,
        'width': source_image_data['imageWidth'],
        'height': source_image_data['imageHeight'],
    })

# Save the results to a JSONL file
with open(output_jsonl_path, 'w') as f:
    for result in results:
        json_line = json.dumps(result, ensure_ascii=False)  # Convert to JSON string
        f.write(json_line + '\n')  # Write to file with a newline

print(f"Results have been saved to {output_jsonl_path}")
