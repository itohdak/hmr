# python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
import cv2

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

import cv_bridge
import rospy
import message_filters
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')

class PeopleMeshDetector(ConnectionBasedTransport):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.sess = tf.Session()
        self.model = RunModel(config, sess=self.sess)

        self.publisher = self.advertise("~output", Image, queue_size=1)

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        # sub_img = message_filters.Subscriber(
        #     '~input', Image, queue_size=1, buff_size=2**24)
        sub_img = message_filters.Subscriber(
            '/usb_cam/image_raw', Image, queue_size=1, buff_size=2**24)
        self.subs = [sub_img]
        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self._cb)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _cb(self, img_msg):
        br = cv_bridge.CvBridge()
        start = rospy.Time.now()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        json_path = None
        input_img, proc_param, img = self._preprocess_image(
            None, img, json_path)
        input_img = np.expand_dims(input_img, 0)
        end = rospy.Time.now()
        print(end - start)
        joints, verts, cams, joints3d, theta = self.model.predict(
            input_img, get_theta=True)
        end = rospy.Time.now()
        print(end - start)

        print(len(joints))
        # print(joints)
        print(joints3d)
        ret_img = self._visualize(img, proc_param, joints[0], verts[0], cams[0])

        # cv2.imshow('camera capture', ret_img)
        pub_img = br.cv2_to_imgmsg(ret_img, encoding='8UC4')
        pub_img.header = img_msg.header

        self.publisher.publish(pub_img)


    def _visualize(self, img, proc_param, joints, verts, cam):
        """
        Renders the result in original image coordinate frame.
        """
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_param, verts, cam, joints, img_size=img.shape[:2])

        # Render results
        skel_img = vis_util.draw_skeleton(img, joints_orig)
        rend_img_overlay = renderer(
            vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
        rend_img = renderer(
            vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
        rend_img_vp1 = renderer.rotated(
            vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
        rend_img_vp2 = renderer.rotated(
            vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])
        return rend_img_overlay

        import matplotlib.pyplot as plt
        # plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(231)
        plt.imshow(img)
        plt.title('input')
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(skel_img)
        plt.title('joint projection')
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(rend_img_overlay)
        plt.title('3D Mesh overlay')
        plt.axis('off')
        plt.subplot(234)
        plt.imshow(rend_img)
        plt.title('3D mesh')
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(rend_img_vp1)
        plt.title('diff vp')
        plt.axis('off')
        plt.subplot(236)
        plt.imshow(rend_img_vp2)
        plt.title('diff vp')
        plt.axis('off')
        plt.draw()
        plt.show()
        # import ipdb
        # ipdb.set_trace()


    def _preprocess_image(self, img_path=None, img=None, json_path=None):
        if img_path:
            img = io.imread(img_path)

        if json_path is None:
            scale = 1.
            center = np.round(np.array(img.shape[:2]) / 2).astype(int)
            # image center in (x,y)
            center = center[::-1]
        else:
            scale, center = op_util.get_bbox(json_path)

        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                   config.img_size)

        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        return crop, proc_param, img


def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret is None:
            break

        input_img, proc_param, img = preprocess_image(
            None, frame, json_path)
        input_img = np.expand_dims(input_img, 0)
        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)
        print(len(joints))
        # print(joints)
        print(joints3d)
        ret = visualize(img, proc_param, joints[0], verts[0], cams[0])

        cv2.imshow('camera capture', ret)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    rospy.init_node('people_mesh_detector')
    PeopleMeshDetector()
    # main(config.img_path, config.json_path)
    rospy.spin()
