"""
This submodule contains data preprocessing functions called in main.py

Includes functions to:
    - expand dimensionality (necessary for tiling)
    - tile WSI (whole slide image) by naive sliding window tiling
    - OR produce patches via finding contours/centroids
    - build MIL bags from tiles/instances
    - one-hot-encode labels
    - convert bags to batches for NN processing
    - histogram equalization
    - histogram matching (can equalize tiles due to differences in laser stability)
    - discard uninformative tiles [via Otsu's thresholding] (containing to much background etc.)
    - standardization (z-scoring) 
    - normalization (min-max scaling, shifts values into [0-1] range)
"""

# IMPORTS
#########

import os
import numpy as np
import cv2  # opencv instead of plt routines
import math


# FUNCTION DEFINITIONS
######################


def expand_dimensionality(data):
    """Takes an list image input[n[x,y,z] ]
    and returns a list of images with shape [1,x,y,z]
    """
    expanded_data = []

    for i in data:
        expanded_image = tf.expand_dims(i, 0)
        print(expanded_image.shape)
        expanded_data.append(expanded_image)

    return expanded_data


def crop_center(img, cropx, cropy):
    """crop center area of patched images constructed in
    blowup_patches.
    E.g. [1,224,224] -> [1,25,25]
    """
    if len(img.shape) == 3:  # grayscale
        y, x = img.shape[1:]
    elif len(img.shape) == 4:  # rgb images
        y, x, _ = img.shape[1:]
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[:, starty : starty + cropy, startx : startx + cropx]


def filter_patches(patch_collection, crop=50, factor=1.0):
    all_patch_means = []
    for i in patch_collection:
        i = crop_center(i, crop, crop)
        patch_mean = i.mean()
        all_patch_means.append(patch_mean)

    all_patch_mean = np.mean(all_patch_means)

    filtered_patch_collection = []
    for img, mean in zip(patch_collection, all_patch_means):
        if mean > (factor * all_patch_mean):
            filtered_patch_collection.append(img)
    if len(filtered_patch_collection) > 0:
        filtered_patch_collection = np.stack(filtered_patch_collection, axis=0)
        return filtered_patch_collection
    else:
        del filtered_patch_collection


def central_cutout(img, size=(1000, 1000)):
    """selects the central region of a 2D array and produces a cutout of it"""
    img_size = img.shape
    x_centroid = int(img_size[0] / 2)
    y_centroid = int(img_size[1] / 2)

    cutout = img[
        x_centroid - int(size[0] / 2) : x_centroid + int(size[0] / 2),
        y_centroid - int(size[1] / 2) : y_centroid + int(size[1] / 2),
    ]
    return cutout


def tile_wsi(image, tile_size=(300, 300), strides=(150, 150)):
    """Tiles an input image(s) into patches of a chosen size
    uses sklearn.feature_extraction.image.extract_patches_2d

    """
    if len(image[0].shape) == 2 or len(image[0].shape) == 3:
        result = []
        for i in image:
            img_shape = i.shape
            intermediate_result = []
            for j in range(int(math.ceil(img_shape[0] / (strides[1] * 1.0)))):
                inter_intermediate_result = []

                for k in range(int(math.ceil(img_shape[1] / (strides[0] * 1.0)))):
                    tile = i[
                        strides[1]
                        * j : min(strides[1] * j + tile_size[1], img_shape[0]),
                        strides[0]
                        * k : min(strides[0] * k + tile_size[0], img_shape[1]),
                    ]
                    if tile.shape == tile_size:
                        if len(image[0].shape) == 2:
                            tile = np.expand_dims(
                                tile, axis=0
                            )  # add channel dimension for torch (only for grayscale images)
                        tile = np.expand_dims(
                            tile, axis=0
                        )  # add batch dimension for concat
                        intermediate_result.append(tile)

                if len(inter_intermediate_result) > 1:
                    inter_intermediate_result = np.concatenate(
                        inter_intermediate_result
                    )
                    print(inter_intermediate_result.shape)

            if len(intermediate_result) > 1:
                intermediate_result = np.concatenate(intermediate_result)
                print(intermediate_result.shape)

            result.append(intermediate_result)

        return result


def zscore(img, axis=(0, 1, 2)):
    """Also often called standardization, which transforms the data into a
    distribution with a mean of 0 and a standard deviation of 1.
    Each standardized value is computed by subtracting the mean of the corresponding feature
    and then dividing by the std dev.
    X_zscr = (x-mu)/std
    """
    mean = np.mean(img, axis=axis, keepdims=True)
    std = np.sqrt(((img - mean) ** 2).mean(axis=axis, keepdims=True))
    return (img - mean) / std


def min_max_norm(img):
    """Also often called normalization, which transforms the data into a range
    [0-1].
    """
    return (img - img.min()) / (img.max() - img.min())


def histogram_equalization(img, mode="normal"):
    """Hist. equal. increases contrast in images by detecting the distribution
    of pixel densities in an image and plotting these on a hist.
    The distr. of this hist. is then analyzed and if there are ranges of pixel brightnesses
    that aren't currently being utilized the histogram is then strethced to cover those ranges
    and is then back-projected onto the image.

    Input:
        a grayscale image
        a mode argument (choose between normal or CLAHE (adaptive))
    Outputs:
        Returns the histogram equalized image (still grayscale)
    """
    if mode == "normal":
        return cv2.equalizeHist(img)
    elif mode == "adaptive":
        # performs Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)


def _match_cumulative_cdf(source, template):
    """Return mofified source array so that the cumulative density function (CDF)
    of its values mathces the CDF of the template.

    Used in histogram_matching
    Reference implementation as found in skimage
    """
    src_values, src_unique_indices, src_counts = np.unique(
        source.ravel(), return_inverse=True, return_counts=True
    )

    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quanitles = np.cumsum(src_counts) / source.size
    tmpl_quanitles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quanitles, tmpl_quanitles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def histogram_matching(img, reference, multichannel=False):
    """Modifies the constrast of an image based on a reference image.
    This is useful to unify the contrast level of a group of images (eg. laser stability fluctuations).
    If there are multiple channels, each channel is matched independently.

    Input:
        a grayscale (or multimodal) image
        a reference image to be matched
        multichannel: bool, optional (if true, apply matching seperatly for each channel)
    Outputs:
        Returns the histogram matched image
    """
    if img.ndim != reference.ndim:
        raise ValueError("Image and reference must have the same number of channels")

    if multichannel:
        if img.shape[-1] != reference.shape[-1]:
            raise ValueError(
                "Number of channels in input image and reference image do not match!"
            )
        matched = np.empty(img.shape, dtype=img.dtype)
        for channel in range(img.shape[-1]):
            matched_channel = _match_cumulative_cdf(
                img[..., channel], reference[..., channel]
            )
            matched[..., channel] = matched_channel

    else:  # grayscale images
        matched = _match_cumulative_cdf(img, reference)

    return matched


def otsu_thresholding(img, blurring=True):
    """Compute a thresholded image according to Otsu's method or
    Otsu's Binarization.
    Uses Gaussian blurring by default to improve the thresholding result.
    Needs a 2D image without z-dimensionality (eg. no channels).
    Most often this is a x*y grayscale image.
    """
    if blurring == True:
        img = cv2.GaussianBlur(
            img.astype("uint8"), (5, 5), 0
        )  # filters image with a Gaussian 5x5 kernel to remove noise
    _, thresholded_img = cv2.threshold(
        img.astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded_img


def sort_contours(cnts, method="left-to-right"):
    """Sort all found contours in an image based on a passed
    method argument.
    """
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def filter_contours(contours, hierarchy):
    """Filters all found contours of an image based on their computed
    tree hierachy.
    eg. compute all parent contours ('outermost' contours)
    """
    print(f"There is a total of {len(contours)} contours.")

    parent_cnts = []
    for cnt, tree in zip(contours, hierarchy.squeeze()):
        if tree[-1] == [-1]:
            parent_cnts.append(cnt)

    print(f"After filtering there are now {len(parent_cnts)} contours.")
    return parent_cnts


def find_contours_and_centroids(
    thresh, morph_kernel=(8, 8), lower_bound_area=1000, upper_bound_area=20000
):
    """applies morphological closing (useful to better segment the singlecells in SRS images)
    Then computes the contours, contour areas and centroids
    Centroids that aren't (0,0) are returned as a list as the selected patches
    """
    # first apply morphological closing to otsu thresh img to better determine cell contours
    kernel = np.ones(morph_kernel, np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )[
        -2:
    ]  # the cluster opencv version has different returns, thus using [-2:] to get the same outputs
    # filter contours (eg. only parents)
    contours = filter_contours(contours, hierarchy)
    # compute centroids
    centroids = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # calculate contour area and limit it between two thresholds
        area = cv2.contourArea(c)
        if (int(area) > lower_bound_area) and (int(area) < upper_bound_area):
            centroids.append(tuple((cX, cY)))

    # delete centroids that are (0,0)
    centroids = list(filter(lambda elem: elem != (0, 0), centroids))
    return centroids


def blowup_patches(img, centroids, patch_size=(224, 224), multichannel=False):
    """locates computed centroids in the original image and extracts patches
    (within the borders) of the image of a given size.
    Also extracts the coordinates of each patch.
    E.g. if patch size is set as (200,200) then the patch will range from x-100 to x+100 and
    y-100 to y+100

    Returns the ndarray as a patch collection of shape (n, patch_size[0], patch_size[1])
    as well as the list of coordinates.
    """
    image_patch_collection = []
    coords = []
    patch_x = patch_size[0]
    patch_y = patch_size[1]

    for point in centroids:
        x, y = point
        # cv2.circle(img, (x, y), 5, (255, 255, 255), -1) # only for debugging purposes to visualize centroid in tiles
        x_left = x - int(patch_x / 2)
        x_right = x + int(patch_x / 2)
        y_left = y - int(patch_y / 2)
        y_right = y + int(patch_y / 2)
        # get patch if it is fully contained in the image
        if (x_left > 0 and y_left > 0) and (
            x_right < img.shape[1] and y_right < img.shape[0]
        ):
            #
            # ATTENTION!
            # since rows represent the y axis and columns are the x axis (in OpenCV), x and y need to "reversed" as below
            #
            patch = img[y_left:y_right, x_left:x_right]
            # patch =  patch.astype('float32') / 255.0
            if not multichannel:
                patch = np.expand_dims(
                    patch, axis=0
                )  # add channel dimension for torch (for grayscale images only)
            image_patch_collection.append(patch)
            coords.append([y_left, y_right, x_left, x_right])

    if len(image_patch_collection) > 0:
        image_patch_collection = np.stack(image_patch_collection, axis=0)
        return image_patch_collection, coords
    else:
        del image_patch_collection


def discard_background_tiles(tiled_data, threshold_lower=0.05, threshold_upper=0.2):
    """Discards all tiles from the tiled data that
    exceed a certain threshold of background pixels
    """
    tiled_data_without_background = []

    for tile_collection in tiled_data:
        placeholder_tile_collection = []

        for tile in tile_collection:
            # tile_2d = tile.mean(axis=-1) # needed for thresholding in multichannel images
            tile_2d = tile.squeeze()
            thresholded_img = otsu_thresholding(tile_2d)

            total_pixels = thresholded_img.shape[0] * thresholded_img.shape[1]
            foreground_pixels = np.count_nonzero(thresholded_img == 255)
            ratio = foreground_pixels / total_pixels
            if (ratio >= threshold_lower) and (ratio <= threshold_upper):
                placeholder_tile_collection.append(tile)
            else:
                continue
        tiled_data_without_background.append(np.stack(placeholder_tile_collection))

    return tiled_data_without_background


def build_bags(tiles, labels):
    """Builds bags suited for MIL problems: A bag is a collection of a variable number of instances. The instance-level labels are not known.
    These instances are combined into a single bag, which is then given a supervised label eg. patient diagnosis label when the instances are multiple tissue instances from that same patient.

    Inputs:
        Data tiled from images with expanded dimensionality, see preprocessing.tile_wsi and .expand_dimensionality

    Outputs:
        Returns two arrays: bags, labels where each label is sorted to a bag. Number of bags == number of labels
        bag shape is [n (tiles,x,y,z) ]
    """
    result_bags = tiles
    result_labels = []
    count = 0

    print(len(result_bags))
    print(len(result_bags[0]))
    print(result_bags[0][0].shape)

    # check number of bags against labels
    if len(result_bags) == len(labels):
        pass

    else:
        raise ValueError(
            "Number of Bags is not equal to the number of labels that can be assigned.\nCheck your input data!"
        )

    # this step seems to be necessary in Tensorflow... it is not possible to use one bag - one label
    for j in labels:
        number_of_instances = result_bags[count].shape[0]
        tiled_instance_labels = np.tile(labels[count], (number_of_instances, 1))
        result_labels.append(tiled_instance_labels)
        count += 1

    return result_bags, result_labels, labels


def convert_bag_to_batch(bags, labels):
    """Convert bag and label pairs into batch format
    Inputs:
        a list of bags and a list of bag-labels

    Outputs:
        Returns a dataset (list) containing (stacked tiled instance data, bag label)
    """
    dataset = []

    for index, (bag, bag_label) in enumerate(zip(bags, labels)):
        batch_data = np.asarray(bag, dtype="float32")
        batch_label = np.asarray(bag_label, dtype="float32")
        dataset.append((batch_data, batch_label))

    return dataset


def one_hot_encode_labels_sk(labels):
    """Takes integer labels with values [0-9] and converts them to one hot encoded labels.
    Uses sklearn instead of keras!
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    labels_transformed = mlb.fit_transform(labels)
    print("Original classes are : " + str(mlb.classes_))
    print("One-hot-encoded classes are : " + str(np.unique(labels_transformed, axis=0)))

    return labels_transformed
