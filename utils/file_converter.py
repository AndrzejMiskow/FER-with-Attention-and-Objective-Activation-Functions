import cv2, os

base_path = "../data/convert/jaffedbase/"
new_path = "../data/convert/converted/"

for infile in os.listdir(base_path):
    print("file : " + infile)
    read = cv2.imread(base_path + infile)
    outfile = '.'.join(infile.split('.')[:3])
    outfile = outfile + '.jpg'
    cv2.imwrite(new_path + outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
