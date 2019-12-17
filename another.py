import PIL
# imports Python Imaging Library for opening, manipulating, saving image formats.
from PIL import Image
# import image module from PIL
import matplotlib.pyplot as plt
# collection of command style functions that make matplotlib work as MATLAB.
import random
# imports random number generation module
import numpy as np
# adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import time
import os
import skimage.io as io
from copy import deepcopy
size = 20


def compress(image, basewidth):
    # function to compress the image
    # image size -> The requested size in pixels, as a 2-tuple: (width, height) -> size[0] is width and size[1] is height
    ratio = (basewidth / float(image.size[0]))
    height_ratio_size = int((float(image.size[1]) * float(ratio)))
    resized = image.resize((basewidth, height_ratio_size), PIL.Image.ANTIALIAS)
    # Returns a resized copy of the image
    # PIL.Image.ANTIALIAS - a high-quality downsampling filter
    return resized


def decToBin(dec):
    # function to convert decimal to binary keep leading zeroes
    # format() method returns a formatted representation of the given value controlled by the format specifier
    # In total 8 bits
    return format(dec, '08b')


def binToDec(binStr):
    # function to convert binary to Decimal
    # int() method converts the specified value in given base into an integer number.
    # Given base is 2 in this case.
    return int(binStr, 2)


def convertBitsToImage(bitStream):
    # function to convert bits to images
    count = 0
    # counter for iteration
    byte = ''
    # 1 byte = 8 bits
    list_of_decimals = []
    # list to store decimals
    for bit in bitStream:
        if type(bit) == str:
            byte += bit
        else:
            byte += str(bit)
        # for each bit in the bitstream check the type: if string -> concatenate it with byte else convert type into string and concatenate it
        count = count + 1
        # 1 bit -> + 1 count
        if count == 8:
            # when 8 bits -> 1 byte, convert byte to decimal value and append it to list of decimals
            list_of_decimals.append(binToDec(byte))
            byte = ''
            count = 0
            # after, make byte empty and counter 0

    length = int(len(list_of_decimals) / 3)

    # find length in order to divide list into red, green and blue channels

    red = list_of_decimals[:length]
    # red : from 0 to length
    green = list_of_decimals[length:(length * 2)]
    # green : from length to length * 2
    blue = list_of_decimals[length * 2:length * 3]
    # blue : from length * 2 to length * 3

    RGB_received = []
    for i in range(length):
        RGB_received.append((red[i], green[i], blue[i]))
        # Each index of the RED GREEN BLUE lists append as a RGB values into RGD_received

    Image_received = np.empty((height, width, 3), dtype=np.uint8)
    # Numpy.empty() - Returns a new array of given shape and type, without initializing entries
    # where (height, width, 3) is a shape
    # 3 -> red blue green values
    # np.unit8 - Byte (-128 to 127)
    x = 0
    y = 0

    for pixel in RGB_received:
        # for each pixel, assign it to new image array size(height, width)
        Image_received[y, x] = pixel
        x = x + 1
        # for each width, 1 is added till it covers the full width
        if x == width:
            y = y + 1
            x = 0
        # when it reaches it, make width 0, increase by 1 the height of the image

    return Image.fromarray(Image_received)

    # fromarray() method creates an image memory from an object exporting the array interface(using the buffer protocol)


# begining of main part of program
img = Image.open("n.jpg")
# compressing size of image so that calculations will be easier
compressed_img = compress(img, 300)
# assigning width and height of image to some variables so that we can use it later for image manipulation
(width, height) = compressed_img.size

# compressed_img.getdata() returns each pixel values of the image
pixel_values = list(compressed_img.getdata())
# here below we're unpacking values of pixel values to red, green, blue color pixels
red, green, blue = zip(*pixel_values)
##__________ Start of RGB _______________##

# extracting Red, Green, Blue channels
red_channel = [(d[0], 0, 0) for d in pixel_values]
green_channel = [(0, d[1], 0) for d in pixel_values]
blue_channel = [(0, 0, d[2]) for d in pixel_values]

#showing
compressed_img.putdata(red_channel)
compressed_img.show()

compressed_img.putdata(green_channel)
compressed_img.show()

compressed_img.putdata(blue_channel)
compressed_img.show()


### END of RGB #####

pixel_values_iterable = red + green + blue
bitStream = ''  # variable for containing bits for transmission
# converting each pixel values from decimal to binary
for pixel_in_decimal in pixel_values_iterable:
    bitStream += decToBin(pixel_in_decimal)

batch_size = width * 24
SNR_range = range(-15, 10, 5)
error_rates = []

for SNR_db in SNR_range:
    starting_time = time.time()
    print(f"--- Noise with SNR has being created: {SNR_db}---")

    start_of_batch = 0
    bit_stream_array = []
    recieved_signal = []
    # start iteration over height in step of batch size
    for h in range(height):
        # adjusting end of batch and starting point in following 3 lines
        end_of_batch = start_of_batch + batch_size
        batch_of_bit_stream = bitStream[start_of_batch: end_of_batch]
        start_of_batch = end_of_batch

        # converting bits in string to int and appending them to list
        single_polar_arr = [int(bit) for bit in batch_of_bit_stream]
 
        bit_stream_array.extend(single_polar_arr)
        single_polar_arr = np.array(single_polar_arr, dtype=np.dtype('i1'))  # making list to numpy array

        bipolar = 2 * single_polar_arr - 1  # formula for converting unipolar values to bipolar
        num_of_bits = len(bipolar)
        sample_len = 2 * np.pi

        sample_range = np.linspace(0, sample_len, size)
        x = np.linspace(0, sample_len * num_of_bits, size * num_of_bits)

        bipolar_digital_sig = []
        for i in range(num_of_bits):
            for j in sample_range:
                bipolar_digital_sig.append(bipolar[i])

        carrier_sig = np.sin(x)  # carrier signal
        modulated_sig = carrier_sig * bipolar_digital_sig

        SNR = 10 ** (SNR_db / 10)  # calculating SNR from SNR_db(SNR in decibal)
        std = np.sqrt(1 / (2 * SNR))  # calculating standard deviation
        noise = np.random.normal(loc=0, scale=std, size=len(x))
        noise_sig = modulated_sig + noise # adding noise to sent signal over channel

        # Demodulation has started here
        demodulated_signal = noise_sig * carrier_sig

        for i in range(0, len(demodulated_signal), size):
            sm = sum(demodulated_signal[i:i + size])
            if sm > 0:
                recieved_signal.append(1)
            else:
                recieved_signal.append(0)

    np_recieved_sig = np.array(recieved_signal, dtype=np.dtype('i1'))
    np_bit_stream = np.array(bit_stream_array, dtype=np.dtype('i1'))

    error_rates.append((np_recieved_sig != np_bit_stream).sum() / np_recieved_sig.size)
    # Converting bits to image
    recieved_img = convertBitsToImage(recieved_signal)
    current_dir = os.getcwd()
    recieved_img.save(current_dir + str(SNR_db) + ".jpg")
    recieved_img.show()
    print("--End of loop: {0:.2f} mins--".format((time.time() - starting_time) / 60))

plt.title("Noise Rate vs. Bit Error Rate")
plt.semilogy(SNR_range, error_rates)
plt.xlabel("Noise Rate")
plt.ylabel("Bit Error Rate")
plt.show()
