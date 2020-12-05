import cv2
import numpy as np
img = cv2.imread("./cat.png")


def bilinear_interpolation_v1(src, dst_size):
    src_h, src_w, channel = src.shape
    dst_w, dst_h = dst_size
    x_scale = dst_w / src_w
    y_scale = dst_h / src_h
    dst = np.zeros((dst_h, dst_w, channel), dtype=src.dtype)
    for c in range(channel):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                src_x = (dst_x + 0.5) / x_scale - 0.5
                src_y = (dst_y + 0.5) / y_scale - 0.5
                src_x1 = int(src_x)
                src_y1 = int(src_y)
                src_x2 = src_x1 + 1
                src_y2 = src_y1 + 1
                def clip(v, vmin, vmax):
                    v = v if v >= vmin else vmin
                    v = v if v <= vmax else vmax
                    return v
                
                src_x1 = clip(src_x1, 0, src_w-1)
                src_x2 = clip(src_x2, 0, src_w-1)
                src_y1 = clip(src_y1, 0, src_h-1)
                src_y2 = clip(src_y2, 0, src_h-1)
                y1_value = (src_x - src_x1) * src[src_y1, src_x2, c] + (src_x2 - src_x) * src[src_y1, src_x1, c]
                y2_value = (src_x - src_x1) * src[src_y2, src_x2, c] + (src_x2 - src_x) * src[src_y2, src_x1, c]
                dst[dst_y, dst_x, c] = (src_y - src_y1) * y2_value + (src_y2 - src_y) * y1_value
    return dst


def bilinear_interpolate(source, scale=2, pad=0.5):
	sour_shape = source.shape
	(sh, sw) = (sour_shape[-2], sour_shape[-1])
	padding = pad*np.ones((sour_shape[0], sour_shape[1], sh+1, sw+1))
	padding[:,:,:-1,:-1] = source

	(th, tw) = (round(scale*sh), round(scale*sw))

	grid = np.array(np.meshgrid(np.arange(th), np.arange(tw)), dtype=np.float32)
	xy = np.copy(grid)
	xy[0] *= sh/th
	xy[1] *= sw/tw
	x = xy[0].flatten()
	y = xy[1].flatten()

	clip = np.floor(xy).astype(np.int)
	cx = clip[0].flatten()
	cy = clip[1].flatten()

	f1 = padding[:,:,cx,cy]
	f2 = padding[:,:,cx+1,cy]
	f3 = padding[:,:,cx,cy+1]
	f4 = padding[:,:,cx+1,cy+1]

	a = cx+1-x
	b = x-cx
	c = cy+1-y
	d = y-cy

	fx1 = a*f1 + b*f2
	fx2 = a*f3 + b*f4
	fy = c*fx1 + d*fx2
	fy = fy.reshape(fy.shape[0],fy.shape[1],tw,th).transpose((0,1,3,2))
	return fy

import time
t1 = time.time()
img_b = bilinear_interpolation_v1(img, (512, 512))
t2 = time.time()

path_list = ['./cat.png']
imgs = []
for path in path_list:
	im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)/255
	imgs.append(img)
imgs = np.array(imgs).transpose((0,3,1,2))
img_c = bilinear_interpolate(imgs)
t3 = time.time()
print("version 1 : cost %6f seconds"%(t2-t1))
print("version 2 : cost %6f seconds"%(t3 - t2))

