def pltImg(images, labels):
    """
    input: 
        images you want to display
        labels
    this function displays 20 images from the head of images.
    """
    images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(labels[idx])