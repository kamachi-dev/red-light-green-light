# import cv2
# import numpy as np
# import mss
# import pygetwindow as gw

# # Find the window
# window = gw.getWindowsWithTitle("Minecraft")[0]

# # Get position
# x, y, width, height = window.left, window.top, window.width, window.height

# sct = mss.mss()
# monitor = {"top": y, "left": x, "width": width, "height": height}

# while True:
#     frame = np.array(sct.grab(monitor))
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
#     cv2.imshow("Window Capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()


import pygetwindow as gw

windows = gw.getAllTitles()

for i, title in enumerate(windows):
    if title.strip():  # skip empty titles
        print(f"{i+1}. {title}")