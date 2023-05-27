import cv2
import numpy as np
import random
import os


def crop_and_resize(img, mask):
    # find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get the largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    largest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    # get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    # crop the image and mask
    cropped_img = img[y : y + h, x : x + w]
    cropped_mask = mask[y : y + h, x : x + w]
    # resize the cropped image and mask
    scale = random.uniform(0.5, 1.5)
    new_size = (int(w * scale), int(h * scale))
    resized_img = cv2.resize(cropped_img, new_size)
    resized_mask = cv2.resize(cropped_mask, new_size)
    return resized_img, resized_mask


def paste_image(img1, img2, mask1, is_seamless=False):
    # get the resized image and mask
    resized_img, resized_mask = crop_and_resize(img1, mask1)
    # get the dimensions of the images
    h1, w1, _ = resized_img.shape
    h2, w2, _ = img2.shape
    # check if the sizes are different, resize the larger image to match the size of the smaller image
    if h1 > h2 or w1 > w2:
        img2 = cv2.resize(img2, (w1, h1))
        h2, w2 = h1, w1
        x, y = 0, 0
    else:
        # get a random position to paste the image
        x = random.randint(0, w2 - w1 - 1)
        y = random.randint(0, h2 - h1 - 1)
    # create a mask for the pasted image
    mask2 = np.zeros((h2, w2), dtype=np.uint8)
    mask2[y : y + h1, x : x + w1] = resized_mask
    resized_mask *= 255
    # blend the images together

    if is_seamless:
        blended_img = cv2.seamlessClone(
            resized_img,
            img2,
            resized_mask,
            (x + w1 // 2, y + h1 // 2),
            cv2.NORMAL_CLONE,
        )
    else:
        pos_y, pos_x = np.where(resized_mask)
        new_pos_x, new_pos_y = pos_x + x, pos_y + y
        img2[new_pos_y, new_pos_x] = resized_img[pos_y, pos_x]
        blended_img = img2

    return blended_img


def paste_masked_image(folder_path):
    # load all images in the folder
    img_names = []
    img_basenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_names.append(filename)
            img_basenames.append(os.path.splitext(filename)[0])

    # loop through each image and paste a masked instance from another random image
    for i, img1 in enumerate(img_names):
        # load a random mask from another image
        j = random.randint(0, len(img_names) - 1)
        while j == i:
            j = random.randint(0, len(img_names) - 1)
        image_i = cv2.imread(os.path.join(folder_path, img_names[i]))
        image_j = cv2.imread(os.path.join(folder_path, img_names[j]))
        k = 1
        while os.path.exists(
            os.path.join(folder_path, f"{img_basenames[j]}_mask_{k}.npz")
        ):
            mask_data = np.load(
                os.path.join(folder_path, f"{img_basenames[j]}_mask_{k}.npz")
            )
            mask2 = mask_data["mask_rle"].astype(np.uint8)
            mask_data.close()
            k += 1
            # paste the masked instance onto the current image
            image_i = paste_image(image_j, image_i, mask2)

        # save the result
        new_folder_path = folder_path + "_masked"
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        cv2.imwrite(
            os.path.join(new_folder_path, f"{img_basenames[i]}_masked.png"), image_i
        )


paste_masked_image("./images")
