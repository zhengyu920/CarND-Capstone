from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
# import os

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.classmap = {1: TrafficLight.GREEN, 2: TrafficLight.RED, 3: TrafficLight.YELLOW, 4: TrafficLight.UNKNOWN}
        ssd_inception_sim_model='light_classification/model/sim_ssdv2_frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ssd_inception_sim_model, 'r') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light
            <type 'numpy.ndarray'>
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        max_idx = 4
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)
                min_score_thresh = .50
                # find majority light state
                counter = [0, 0, 0, 0, 0]
                for i in range(boxes.shape[0]):
                    if scores is None or scores[i] > min_score_thresh:
                        counter[classes[i]] += 1
                for i in range(1, 5):
                    if counter[i] > counter[max_idx]:
                        max_idx = i
        return self.classmap[max_idx]
