## Reviving Iterative Training with Mask Guidance for Interactive Segmentation 

### Using ObjSegmentation

Import the class

    from ObjSegmentation import ObjSegmentation

Provide your GT/AG points in the form of a list of dicts with x and y coordinates:

    gt_points = [{'x': 243.21, 'y':957.21}, {...}]
    ag_points = [{'x': 221.73, 'y':946.53}, {...}]

Export the mask of your segmented object created using the GUI

Initialize your object:

    object_1 = ObjSegmentation('image_name.png', gt_points, ag_points, 'mask.png')

To visualize masks:

    plt.imshow(object_1.gt_mask)
    plt.imshow(object_1.agent_mask)
    plt.imshow(object_1.ritm_mask)
    plt.imshow(object_1.grabcut_mask)

To get IoUs, pass in the two masks to the get_iou function. For example:

RITM IoU:

    object_1.get_iou(object_1.gt_mask, object_1.ritm_mask)

Agent IoU:

    object_1.get_iou(object_1.gt_mask, object_1.agent_mask)
