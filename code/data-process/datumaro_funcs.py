from copy import deepcopy
import datumaro as dm
from tqdm import tqdm

from datumaro.util.mask_tools import rle_to_mask, mask_to_polygons
from datumaro.components.annotation import AnnotationType

def remap_label_by_attribute(dataset, original_label, attribute, value, new_label, n_check = 0):
    """
    Remaps a specific label based on an attribute value.
    UPDATED: Now checks if the new label has at least n_check instances before applying changes.

    Parameters:
    - dataset (datumaro.Dataset): The dataset to modify.
    - original_label (str): The original label name to check.
    - attribute (str): The attribute key to filter on.
    - value (str): The attribute value to match.
    - new_label (str): The new label name to assign.
    - n_check (int): The minimum number of instances the new label must have. If less, no changes are applied.

    Returns:
    - dataset (datumaro.Dataset): The modified dataset with remapped label.
    """

    # Extract label categories
    categories = dataset.categories()[dm.AnnotationType.label]
    label_map = {idx: cat.name for idx, cat in enumerate(categories.items)}

    # Ensure new label exists before remapping. Si no, el update no se hace bien
    if new_label not in categories:
        categories.add(new_label)

    # Clone dataset for modification
    filtered_dataset = deepcopy(dataset)

    # Filter dataset to keep only items with the specified attribute condition
    filtered_dataset.filter(
        lambda item: any(
            ann.type.name == "mask" and
            label_map.get(ann.label) == original_label and
            ann.attributes.get(attribute) == value
            for ann in item.annotations
        )
    )

    # Apply remap_labels only to filtered items
    filtered_dataset.transform("remap_labels", mapping={original_label: new_label})
    
    # Check if there are at least n instances of the new label
    instance_dict = count_instance_labels(filtered_dataset)
    if instance_dict[new_label] < n_check:
        print(f"WARNING: New label '{new_label}' has less than {n_check} instances. No changes applied.")
        return dataset
    else:
        # Merge transformed dataset back into original dataset
        dataset.update(filtered_dataset)

        return dataset  # Return modified dataset

# Now transform the masks from RLE to polygons
def convert_rle_to_polygons(dataset):
    """
    Converts all RLE annotations in a dataset to polygon annotations.

    Parameters:
    - dataset (datumaro.Dataset): The dataset to modify.

    Returns:
    - dataset (datumaro.Dataset): The modified dataset with polygon annotations.
    """

    new_items = []

    for item in tqdm(dataset):
        new_annotations = []

        for ann in item.annotations:
            if ann.type == dm.AnnotationType.mask:  # Check if annotation is a mask
                if isinstance(ann.image, dict) and "counts" in ann.image:  # Check if RLE format
                    # Convert RLE to binary mask
                    binary_mask = rle_to_mask(ann.image)
                    
                    # Convert binary mask to polygons
                    polygons = mask_to_polygons(binary_mask)

                    # Create new polygon annotations
                    for poly in polygons:
                        new_annotations.append(
                            dm.Polygon(poly, label=ann.label, attributes=ann.attributes)
                        )
                else:
                    # If it's already a binary mask, just convert it
                    polygons = mask_to_polygons(ann.image)
                    for poly in polygons:
                        new_annotations.append(
                            dm.Polygon(poly, label=ann.label, attributes=ann.attributes)
                        )
            else:
                # Keep other annotations as they are
                new_annotations.append(ann)

        # Create a new dataset item with updated annotations
        new_items.append(item.wrap(annotations=new_annotations))

    # Create a new dataset with transformed annotations
    return dm.Dataset.from_iterable(new_items, categories=dataset.categories())

def remove_categories(dataset, categories):
    """
    Removes specified categories from a dataset.

    Parameters:
    - dataset (datumaro.Dataset): The dataset to modify.
    - categories (list of str): The categories to remove.

    Returns:
    - dataset (datumaro.Dataset): The modified dataset with specified categories removed.
    """

    # Clone dataset for modification
    filtered_dataset = deepcopy(dataset)
    
    categories_dict = {name: '' for name in categories}

    filtered_dataset.transform('remap_labels',
                               mapping = categories_dict,
                               default='keep')

    return filtered_dataset  # Return modified dataset

def count_instance_labels(dataset):
    # Retrieve the label categories from the dataset
    label_categories = dataset.categories().get(AnnotationType.label)

    # Use a dictionary to count instances per label name
    label_instance_counts = {}
    for idx, label_cat in enumerate(label_categories.items):
        label_instance_counts[label_cat.name] = 0       # If we don't initialize all to zero, we don't get data from those categories

    # Iterate over each dataset item and its annotations
    for item in dataset:
        for ann in item.annotations:
            label_id = ann.label
            # Get the label name from label_id
            label_name = label_categories[label_id].name
            label_instance_counts[label_name] += 1
                
    return label_instance_counts