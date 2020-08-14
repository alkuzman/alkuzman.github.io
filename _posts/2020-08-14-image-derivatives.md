---
title: Image Derivatives
---

# Image Derivatives
This exercise introduces image derivative operators.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import ndimage
```

## Some Convenience Functions.


```python
def convolve_with_two(image, kernel1, kernel2):
    """Apply two filters, one after the other."""
    image = ndimage.convolve(image, kernel1)
    image = ndimage.convolve(image, kernel2)   
    return image

def imread_gray(filename):
    """Read grayscale image."""
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)

def plot_multiple(images, titles, colormap='gray', max_columns=np.inf, share_axes=True):
    """Plot multiple images as subplots on a grid."""
    assert len(images) == len(titles)
    n_images = len(images)
    n_cols = min(max_columns, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4),
        squeeze=False, sharex=share_axes, sharey=share_axes)

    axes = axes.flat
    # Hide subplots without content
    for ax in axes[n_images:]:
        ax.axis('off')
        
    if not isinstance(colormap, (list,tuple)):
        colormaps = [colormap]*n_images
    else:
        colormaps = colormap

    for ax, image, title, cmap in zip(axes, images, titles, colormaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        
    fig.tight_layout()
```


```python
# From Question 1: Gaussian Filtering
def gauss(x, sigma):
    ### BEGIN SOLUTION
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(- x**2 / 2.0 / sigma**2)
    ### END SOLUTION
```

## Part a
Implement a function for creating a Gaussian derivative filter in 1D according to the following equation

$$
\begin{eqnarray}
    \frac{d}{dx} G &=& \frac{d}{dx} \frac{1}{\sqrt{2\pi}\sigma} \operatorname{exp}\biggl(-\frac{x^2}{2\sigma^2}\biggr) \\
			       &=& -\frac{1}{\sqrt{2\pi}\sigma^3}x \operatorname{exp}\biggl(-\frac{x^2}{2\sigma^2}\biggr)
\end{eqnarray}
$$

Your function should take a vector of integer values $$x$$ and the standard deviation ``sigma`` as arguments.


```python
def gaussdx(x, sigma):
    ### BEGIN SOLUTION
    return - 1.0 / np.sqrt(2.0 * np.pi) / sigma**3 * x * np.exp(- x**2 / 2.0 / sigma**2)
    ### END SOLUTION
```


```python
x = np.linspace(-5, 5, 100)
y = gaussdx(x, sigma=1.0)
fig, ax = plt.subplots()
ax.plot(x, y)
fig.tight_layout()
```


![png]({{site.baseurl}}/assets/imgs/output_7_0.png)


The effect of a filter can be studied by observing its so-called *impulse response*.
For this, create a test image in which only the central pixel has a non-zero value (called an *impulse*):


```python
### BEGIN SOLUTION
impulse = np.zeros((25, 25), dtype=np.float32)
impulse[12, 12] = 255
### END SOLUTION
```

Now, we create the following 1D filter kernels ``gaussian`` and ``derivative``.


```python
sigma = 6.0
kernel_radius = int(3.0 * sigma)
x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
G = gauss(x, sigma)
D = gaussdx(x, sigma)
```

What happens when you apply the following filter combinations?

- first ``gaussian``, then ``gaussian^T``.
- first ``gaussian``, then ``derivative^T``.
- first ``derivative``, then ``gaussian^T``.
- first ``gaussian^T``, then ``derivative``.
- first ``derivative^T``, then ``gaussian``.

Display the result images with the `plot_multiple` function. Describe your result.


```python
images = [
    impulse,
    convolve_with_two(impulse, G, G.T),
    convolve_with_two(impulse, G, D.T),
    convolve_with_two(impulse, D, G.T),
    convolve_with_two(impulse, G.T, D),
    convolve_with_two(impulse, D.T, G)]

titles = [
    'original',
    'first G, then G^T',
    'first G, then D^T',
    'first D, then G^T',
    'first G^T, then D',
    'first D^T, then G']

plot_multiple(images, titles, max_columns=3)
```


![png]({{site.baseurl}}/assets/imgs/output_13_0.png)


## Part b

Use the functions ``gauss`` and ``gaussdx`` directly in order to create a new function ``gaussderiv`` that returns the 2D Gaussian derivatives of an input image in $x$ and $y$ direction.


```python
def gauss_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    ### BEGIN SOLUTION
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    
    image_dx = convolve_with_two(image, D, G.T)
    image_dy = convolve_with_two(image, G, D.T)
    ### END SOLUTION
    return image_dx, image_dy
```

Try the function on the given example images and describe your results.


```python
image = imread_gray({{site.baseurl}}/assets/imgs/'tomatoes.png')
grad_dx, grad_dy = gauss_derivs(image, sigma=5.0)
plot_multiple([image, grad_dx, grad_dy], ['Image', 'Derivative in x-direction', 'Derivative in y-direction'])
```


![png]({{site.baseurl}}/assets/imgs/output_17_0.png)


In a similar manner, create a new function ``gauss_second_derivs`` that returns the 2D second Gaussian derivatives $\frac{d^2}{dx^2}$, $\frac{d^2}{dx dy}$ and $\frac{d^2}{dy^2}$ of an input image.


```python
def gauss_second_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    ### BEGIN SOLUTION
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    
    image_dx, image_dy = gauss_derivs(image, sigma)
    
    image_dxx = convolve_with_two(image_dx, D, G.T)
    image_dyy = convolve_with_two(image_dy, G, D.T)
    image_dxy = convolve_with_two(image_dx, G, D.T)
    ### END SOLUTION
    
    return image_dxx, image_dxy, image_dyy
```

Try the function on the given example images and describe your results.


```python
image = imread_gray('coins1.jpg')
grad_dxx, grad_dxy, grad_dyy = gauss_second_derivs(image, sigma=2.0)
plot_multiple([image, grad_dxx, grad_dxy, grad_dyy], ['Image', 'Dxx', 'Dxy','Dyy'])
```


![png]({{site.baseurl}}/assets/imgs/output_21_0.png)



```python
image = imread_gray('circuit.png')
grad_dxx, grad_dxy, grad_dyy = gauss_second_derivs(image, sigma=2.0)
plot_multiple([image, grad_dxx, grad_dxy, grad_dyy], ['Image', 'Dxx', 'Dxy','Dyy'])
```


![png]({{site.baseurl}}/assets/imgs/output_22_0.png)


## Part c
Create a new function ``image_gradients_polar`` that returns two images with the magnitude and orientation of the gradient for each pixel of the input image.


```python
def image_gradients_polar(image, sigma):
    ### BEGIN SOLUTION
    grad_dx, grad_dy = gauss_derivs(image, sigma)
    magnitude = np.sqrt(grad_dx**2.0 + grad_dy**2.0)
    direction = np.arctan2(grad_dy, grad_dx)
    ### END SOLUTION
    return magnitude, direction
```

Try the function on the given example images and describe your results.


```python
image = imread_gray('coins1.jpg')
grad_mag, grad_dir = image_gradients_polar(image, sigma=2.0)

# Note: the twilight colormap only works since Matplotlib 3.0, use 'gray' in earlier versions.
plot_multiple([image, grad_mag, grad_dir], ['Image', 'Magnitude', 'Direction'], colormap=['gray', 'gray', 'twilight']) 
```


![png]({{site.baseurl}}/assets/imgs/output_26_0.png)



```python
image = imread_gray('circuit.png')
grad_mag, grad_theta = image_gradients_polar(image, sigma=2.0)
plot_multiple([image, grad_mag, grad_theta], ['Image', 'Magnitude', 'Direction'], colormap=['gray', 'gray', 'twilight'])
```


![png]({{site.baseurl}}/assets/imgs/output_27_0.png)


## Part d
Create a new function ``laplace`` that returns an image with the Laplacian-of-Gaussian for each pixel of the input image.


```python
def laplace(image, sigma):
    ### BEGIN SOLUTION
    grad_dxx, _, grad_dyy = gauss_second_derivs(image, sigma)
    return grad_dxx + grad_dyy
    ### END SOLUTION
```

Try the function on the given example images and describe your results.


```python
image = imread_gray('coins1.jpg')
lap = laplace(image, sigma=2.0)
plot_multiple([image, lap], ['Image', 'Laplace'])
```


![png]({{site.baseurl}}/assets/imgs/output_31_0.png)



```python
image = imread_gray('circuit.png')
lap = laplace(image, sigma=2.0)
plot_multiple([image, lap], ['Image', 'Laplace'])
```


![png]({{site.baseurl}}/assets/imgs/output_32_0.png)
