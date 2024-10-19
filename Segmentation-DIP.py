import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# Calculate F1 Score
def calculate_f1_score(ground_image, segmented_image):
    ground_image = ground_image.flatten()
    segmented_image = segmented_image.flatten()
    f1 = f1_score(ground_image, segmented_image, average="weighted")
    return f1


# Display Images
def display_image(original_image, segmented_image):
    plt.subplot(121), plt.imshow(original_image, cmap="gray")
    plt.title("Original Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(segmented_image, cmap="gray")
    plt.title("Segmented Image"), plt.xticks([]), plt.yticks([])
    plt.show()


# Display F1 Scores
def display_f1_scores(
    thresholding_f1, edge_detection_f1, region_based_f1, clustering_f1
):
    plt.bar(
        ["Thresholding", "Edge Detection", "Region Based", "Clustering"],
        [thresholding_f1, edge_detection_f1, region_based_f1, clustering_f1],
    )
    plt.xlabel("Segmentation Technique")
    plt.ylabel("F1 Score")
    plt.title("F1 Score of Different Segmentation Techniques")

    for i, v in enumerate(
        [thresholding_f1, edge_detection_f1, region_based_f1, clustering_f1]
    ):
        plt.text(i, v, str(round(v, 2)), ha="center", va="bottom")

    plt.show()


# Thresholding Segmentation
def thresholding(img):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return thresh


# Edge Detection Segmentation
def edge_detection(img):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    # Calculate lower and upper thresholds for Canny edge detection
    sigma = 0.33
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # Apply Canny edge detection
    edges = cv.Canny(gray, lower, upper)

    # Dilate the edges to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv.dilate(edges, kernel, iterations=2)

    # Clean the edges using morphological opening
    cleaned_edges = cv.morphologyEx(dilated_edges, cv.MORPH_OPEN, kernel, iterations=1)

    # Find connected components and filter out small components
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        cleaned_edges, connectivity=8
    )
    min_size = 500
    filtered_mask = np.zeros_like(cleaned_edges)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255

    # Find contours of the filtered mask
    contours, _ = cv.findContours(
        filtered_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    # Create a filled mask using the contours
    filled_mask = np.zeros_like(gray)
    cv.drawContours(filled_mask, contours, -1, (255), thickness=cv.FILLED)

    # Perform morphological closing to fill holes
    final_filled_image = cv.morphologyEx(
        filled_mask, cv.MORPH_CLOSE, kernel, iterations=2
    )

    # Dilate the final filled image to smooth the edges
    final_filled_image = cv.dilate(final_filled_image, kernel, iterations=1)

    return final_filled_image


# Region Based Segmentation
def region_based(img):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Apply morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Dilate the opening to get the background
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Calculate the distance transform
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

    # Threshold the distance transform to get the foreground
    _, sure_fg = cv.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Find the unknown region
    unknown = cv.subtract(sure_bg, sure_fg)

    # Label the markers for watershed algorithm
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    markers = cv.watershed(img, markers)

    # Create a mask for the segmented region
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers == 1] = 255

    return mask


# Clustering Segmentation
def clustering(img):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Reshape the image
    Z = gray.reshape((-1, 3))
    Z = np.float32(Z)

    # Define the criteria for k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set the number of clusters
    K = 2

    # Perform k-means clustering
    _, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Convert the center values to uint8
    center = np.uint8(center)

    # Reshape the result
    res = center[label.flatten()]
    res = res.reshape((gray.shape))

    # Apply thresholding to the result
    _, res = cv.threshold(res, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return res


def display_single_image_segmentation(num):
    image_path = f"MIVIA Lab/Main_Dataset/Images/{num}/{num}.bmp"

    # Read Image
    img = cv.imread(image_path)

    # Thresholding Segmentation
    thresh = thresholding(img)
    display_image(img, thresh)

    # Edge Detection Segmentation
    edge = edge_detection(img)
    display_image(img, edge)

    # Region Based Segmentation
    region = region_based(img)
    display_image(img, region)

    # Clustering Segmentation
    cluster = clustering(img)
    display_image(img, cluster)


def evaluate_segmentation(segmentation_function):
    path = "MIVIA Lab\Main_Dataset\Images"
    f1_scores = []

    for foldername in os.listdir(path):
        filename = f"{path}\{foldername}\{foldername}.bmp"

        original_image = cv.imread(filename)
        segmented_image = segmentation_function(original_image)
        ground_image = cv.imread(
            f"{path}\{foldername}\{foldername}_mask.bmp", cv.IMREAD_GRAYSCALE
        )

        f1_score = calculate_f1_score(ground_image, segmented_image)
        f1_scores.append(f1_score)

    return np.mean(f1_scores)


def main():
    display_single_image_segmentation("01")

    thresholding_f1 = evaluate_segmentation(thresholding)
    edge_detection_f1 = evaluate_segmentation(edge_detection)
    region_based_f1 = evaluate_segmentation(region_based)
    clustering_f1 = evaluate_segmentation(clustering)

    display_f1_scores(
        thresholding_f1, edge_detection_f1, region_based_f1, clustering_f1
    )


if __name__ == "__main__":
    main()
