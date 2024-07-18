from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import dsproc

img = Image.open("aus.jpg")
data = np.array(img)

# We just want the outline
mask = np.any(data >=200, axis=2)
data[mask] = 0
data[np.invert(mask)] = 255
data = data[:, :, 0]

# Nice! All we have to do is blob them up together
# Here's the plan:
# Pass a kernel over the image
# If the kernel contains nonzero pixels, zero the non-zero pixels and add in a new non-zero pixel between the zero'd
# elements
k_size = 20

for i in range(0, data.shape[1], k_size):

    for j in range(0, data.shape[0], k_size):
        subset = data[j:j+k_size, i:i+k_size]
        # Get the indices of the non_zero elements
        rows, cols = np.where(subset==255)

        # If there is no 1
        if rows.size == 0:
            continue

        # find the average row index and col index and round to nearest int
        row_ind = int(round(rows.mean(), 0))
        col_ind = int(round(cols.mean(), 0))

        # if row_ind > 10:
        #     row_ind = 10
        #
        # if col_ind > 10:
        #     col_ind = 10

        # Zero it all
        data[j:j+k_size, i:i+k_size] *= 0

        # Turn on the correct spot
        # data[j+row_ind:j+row_ind+5, i+col_ind:i+col_ind+5] = 255
        data[j + row_ind, i + col_ind] = 255

# z = Image.fromarray(data)
# z.show()

# Now we just need to convert it to points
y, x = np.where(data != 0)

# midpoints
mid_y = int(data.shape[0]/2)
mid_x = int(data.shape[1]/2)

# Place the points around zero
y -= mid_y
x -= mid_x

# Place on -1 to 1 scale
y = y / max(abs(y))
x = x / max(abs(x))

# We need to flip the y axis because matrices count down but graphs count up
y *= -1

#plt.scatter(x, y)

# Make the constellation!
c = x + 1j*y
# Get every N point
#c = c[::3]

# Now lets make the signal
F = 1000

# Create a random message of n symbols with m levels
MESSAGE = dsproc.utils.create_message(n=10000, m=len(c))
SYMBOL_RATE = 250      # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration (in seconds)

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate (for ease of use reasons)
Fs = SYMBOL_RATE
while Fs <= F * 2:
    Fs += SYMBOL_RATE

s = dsproc.Mod(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)
s.QAM(type="custom", custom_map=c)
s.baseband()
s.iq()

s.save("AUS_Test")


