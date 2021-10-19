import os
import numpy as np

test_images = os.listdir('testing_images/')  # all the testing images

submission = []
for img in test_images:  # image order is important to your result
    predicted_class = your_model(img)  # the predicted category
    submission.append([img, predicted_class])

np.savetxt('answer.txt', submission, fmt='%s')

# access id2name dic as follows
# df_2[df_2['class_id']==0]['class_name'].values[0]

