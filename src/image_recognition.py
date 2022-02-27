import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
import argparse
import imutils

from std_msgs.msg import Int32MultiArray

## get graph
with tf.gfile.FastGFile("./../model/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

## get label
label_lines = [line.rstrip() for line
                   	in tf.gfile.GFile("./../model/output_labels.txt")]

def setup_args():
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  global FLAGS
  FLAGS, unparsed = parser.parse_known_args()
  return unparsed

class RosTensorFlow():
    def __init__(self):
        self._session = tf.Session()

        self._cv_bridge = CvBridge()

        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        #self._pub = rospy.Publisher('result', String, queue_size=1)
        self._pub = rospy.Publisher('result',Int32MultiArray,queue_size = 10)
        self.score_threshold = rospy.get_param('~score_threshold', 0.7)
        self.use_top_k = rospy.get_param('~use_top_k', 5)

        self.result = [-1, -1, -1]
        self.msg = Int32MultiArray()

    def callback(self, image_msg):
        self.result = [-1, -1, -1]

        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
    
        cv_image = cv2.flip(cv_image,-1)
        #cv2.imshow('cv_image', cv_image)

        x=240; y=90; w=90; h=200
        roi = cv_image[y:y+h, x:x+w]    
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
        #roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) 
        cv2.imshow('gray', gray)

        # blur = cv2.GaussianBlur(gray, (7, 7), 0.5)
        # cv2.imshow('blur', blur)

        # thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('thresh', thresh)

        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        # cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)

        # clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8, 8))
        # gray_cont_dst = clahe.apply(gray)
        # cv2.imshow('gray_cont_dst', gray_cont_dst)

        # kernel = np.ones((3, 3), np.uint8)
        # morph = cv2.morphologyEx(gray_cont_dst, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('morph', morph)

        # canny = cv2.Canny(gray_cont_dst, 0, 150, (5,5))
        # cv2.imshow('canny', canny)


        x=260; y=105; w=40; h=40
        roi_1 = cv_image[y:y+h, x:x+w]    
        roi_1 = cv2.cvtColor(roi_1, cv2.COLOR_BGR2GRAY) 
        roi_1 = cv2.cvtColor(roi_1, cv2.COLOR_GRAY2BGR) 
        cv2.imshow('roi_1', roi_1)

        x=265; y=165; w=40; h=40
        roi_2 = cv_image[y:y+h, x:x+w]     
        cv2.imshow('roi_2', roi_2)

        x=270; y=225; w=40; h=40
        roi_3 = cv_image[y:y+h, x:x+w]     
        cv2.imshow('roi_3', roi_3)

        # copy from
        # classify_image.py
        image_data_1 = cv2.imencode('.jpg', roi_1)[1].tostring()
        image_data_2 = cv2.imencode('.jpg', roi_2)[1].tostring()
        image_data_3 = cv2.imencode('.jpg', roi_3)[1].tostring()
        # Creates graph from saved GraphDef.
        softmax_tensor = self._session.graph.get_tensor_by_name('final_result:0')
        predictions_1 = self._session.run(
            softmax_tensor, {'DecodeJpeg/contents:0': image_data_1})
        predictions_2 = self._session.run(
            softmax_tensor, {'DecodeJpeg/contents:0': image_data_2})
        predictions_3 = self._session.run(
            softmax_tensor, {'DecodeJpeg/contents:0': image_data_3})
        predictions_1 = np.squeeze(predictions_1)
        predictions_2 = np.squeeze(predictions_2)
        predictions_3 = np.squeeze(predictions_3)
        # Creates node ID --> English string lookup.
       
        top_k_1 = predictions_1.argsort()[-self.use_top_k:][::-1]
        for node_id in top_k_1:
            human_string = label_lines[node_id]
            score = predictions_1[node_id]
            if score > self.score_threshold:
                #rospy.loginfo('1: %s (score = %.5f)' % (human_string, score))
                #self._pub.publish(human_string)
                self.result[0] = int(human_string)

        top_k_2 = predictions_2.argsort()[-self.use_top_k:][::-1]
        for node_id in top_k_2:
            human_string = label_lines[node_id]
            score = predictions_2[node_id]
            if score > self.score_threshold:
                #rospy.loginfo('2: %s (score = %.5f)' % (human_string, score))
                #self._pub.publish(human_string)
                self.result[1] = int(human_string)

        top_k_3 = predictions_3.argsort()[-self.use_top_k:][::-1]
        for node_id in top_k_3:
            human_string = label_lines[node_id]
            score = predictions_3[node_id]
            if score > self.score_threshold:
                #rospy.loginfo('3: %s (score = %.5f)' % (human_string, score))
                #self._pub.publish(human_string)
                self.result[2] = int(human_string)

        if (self.result[0] != -1) and (self.result[1] != -1) and (self.result[2] != -1):
            rospy.loginfo('1:%s/ 2:%s/ 3:%s' % (str(self.result[0]), str(self.result[1]), str(self.result[2])))

            # track value
            

            self.msg.data = self.result
            self._pub.publish(self.msg)
        cv2.waitKey(3)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    setup_args()
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
