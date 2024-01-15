import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
import urllib.request
from io import BytesIO

# url = str('https://imgur.com/nmGz7Lv.png')
# with urllib.request.urlopen(url) as url:
#     f = BytesIO(url.read())

f = "car.jpg"

print("1. Tiến hành mở bức ảnh")
img = Image.open(f)

# In ra hinh anh ban dau
print("2. Tiến hành vẽ bức ảnh ban đầu")
fig = plt.figure("Hình ảnh ban đầu")
plt.imshow(img)
plt.title("Hình ảnh ban đầu")
print("3. Đã đọc và vẽ xong bức ảnh ban đầu")

# Chuyen sang dinh dang BMP de chuyen sang mang NP, sau do chuyen sang anh xam 
print("4. Tiến hành chuyển đổi sang ảnh xám")
np_img = np.array(img)
np_img_grey = np_img.dot([0.299, 0.5870, 0.114])
fig = plt.figure("Hình ảnh làm xám")
plt.imshow(np_img_grey)
plt.title("Hình ảnh xám")
print("5. Đã biến đổi thành ảnh xám và vẽ xong xám")


# Tinh tich chap 2 chieu
# O day, X la ma tran dau vao (2 chieu, chu nhat)
# F la ma tran loc (2 chieu, vuong)
# s la buoc truot
# Ta se phai tu tinh padding de them vao dau va cuoi cho phu hop
def conv2d(img, filt, slide = 1):
    print("6. Thực hiện hàm tính tích chập")

    img_width, img_height = img.shape
    filt_size = filt.shape[0]

    # O day ta se them padding o hai dau buc anh de chan buoc truot
    # Phan thuc hien tren blog phamdinhkhanh co le chua chinh xac lam
    # Padding o dau va cuoi chua chac da bang nhau (truong hop so du le 
    # thi ta khong the chia deu padding deu o dau va cuoi duoc)
    # Hon nua padding theo chieu dai va chieu cao co the khac nhau
    if (img_width - filt_size) % slide != 0:
        width_pad = slide - (img_width - filt_size) % slide
        width_pad_head = width_pad // 2
        width_pad_tail = width_pad - width_pad_head
    else:
        width_pad = 0
        width_pad_head = 0
        width_pad_tail = 0
    if (img_height - filt_size) % slide != 0:
        height_pad = slide - (img_height - filt_size) % slide
        height_pad_head = height_pad // 2
        height_pad_tail = height_pad - height_pad_head
    else:
        height_pad = 0
        height_pad_head = 0
        height_pad_tail = 0

    res_width = (img_width + width_pad - filt_size) // slide
    res_height = (img_height + height_pad - filt_size) // slide
    res = np.zeros((res_width, res_height))
    
    if width_pad > 0 or height_pad > 0:
        img_pad = np.pad(img, ((width_pad_head, width_pad_tail), (height_pad_head, height_pad_tail)))
    else:
        img_pad = img
    
    for i in range(res_width):
        for j in range(res_height):
            i_orig = i * slide
            j_orig = j * slide
            res[i][j] = np.abs(np.sum(img_pad[i_orig:(i_orig+filt_size), j_orig:(j_orig+filt_size)] * filt))

    return res

# Tao bo loc vi phan detect hang ngang (so sanh hieu giua cac hang)
hor_det_filter = np.array([
    [-1, -1, -1],
    [ 0,  0,  0], 
    [ 1,  1,  1]
])

print("7. Tính tương quan với bộ lọc ngang")
# Tien hanh tinh tich chap voi bo loc detect theo chieu ngang
hor_conv = conv2d(np_img_grey, hor_det_filter, 1)

print("8. Tính xong tương quan với bộ lọc ngang. Tiến hành vẽ kết quả")
fig = plt.figure("Hình ảnh lọc ngang")
plt.imshow(hor_conv)

# Tao bo loc detect cot doc
ver_det_filter = np.array([
    [1, 0, -1], 
    [1, 0, -1], 
    [1, 0, -1]
])

# Tien hanh tinh tich chap voi bo loc detect theo chieu doc
print("9. Tính tương quan với bộ lọc dọc")
ver_conv = conv2d(np_img_grey, ver_det_filter, 3)

print("10. Tính xong tương quan với bộ lọc dọc. Tiến hành vẽ kết quả")
fig = plt.figure("Hình ảnh lọc dọc")
plt.imshow(ver_conv)
print("11. Vẽ kết quả xong. Tiến hành show các hình ảnh lên để xem")

# Show cac hinh anh ra de xem
plt.show()
print("12. Đã show các hình ảnh lên để xem xong. Thoát chương trình")





