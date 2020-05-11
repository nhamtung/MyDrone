from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import datetime
from six.moves import xrange
import glob
from skimage import io, transform, color
import glob
import os
import time
from PIL import Image
import imageio
import pylab
import cv2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU/source/models/", "path to logs directory")
#tf.flags.DEFINE_string("logs_dir", "models/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/home/Water_Detection", "path to dataset")
tf.flags.DEFINE_string("output_dir", "/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU/results/", "path of output")
tf.flags.DEFINE_float("learning_rate", "1e-6", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
#tf.flags.DEFINE_string("model_dir", "content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU/source/models/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
batch_offset = 0
IMAGE_SIZE_HEIGHT = 360
IMAGE_SIZE_WIDTH = 640
NUM_OF_CHANNEL = 3

PATH_IMAGE = "/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU/imageTest"
PATH_VIDEO = "/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU/video/video1.mkv"
PATH_DATASET_TEST = "/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU/Dataset/on_road_test.txt"

def load_training_dataset_path():
  p = np.genfromtxt('/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU/Dataset/on_road_train.txt',dtype='str')
  return p

def load_data(p, step):
    imgs=[]
    gt_imgs=[]
    gt_imgs2=[]
    for i in range(p.shape[0]):
        fp = p[i,0]
        fp = fp.replace("..", "/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU")
        print("TungNV: ", fp)
        fp_gt = p[i,1]
        fp_gt = fp_gt.replace("..", "/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU")
        print("TungNV: ",fp_gt)
        print("Loading images: %s \t %s"%(fp, fp_gt))

        img = io.imread(fp)
        img = img[:,0:1280,:]
        img = transform.resize(img, (IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH))
        imgs.append(img)

        gt_img = io.imread(fp_gt)
        gt_img = transform.resize(gt_img, (IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH))
        for r in range(gt_img.shape[0]):
            for c in range(gt_img.shape[1]):
                if gt_img[r,c] == 1:
                    gt_img[r,c] = 1
                else:
                    gt_img[r,c] = 0
        gt_imgs.append(gt_img)

        gt_img2 = io.imread(fp_gt)
        gt_img2 = transform.resize(gt_img2, (IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH))
        for r in range(gt_img2.shape[0]):
            for c in range(gt_img2.shape[1]):
                if gt_img2[r,c] == 1:
                    gt_img2[r,c] = 0
                else:
                    gt_img2[r,c] = 1
        gt_imgs2.append(gt_img2)

    return np.asarray(imgs,np.float32), np.asarray(gt_imgs,np.int32), np.asarray(gt_imgs2, np.int32)

def load_model(saver, sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    print("TungNV_ckpt: ", ckpt)
    print("TungNV_ckpt.model_checkpoint_path: ", ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Model restored...")
    return ckpt

def inference(image, keep_prob):

    #IMG_MEAN = np.array((104.00698793/255,116.66876762/255,122.67891434/255,146.01657/255), dtype=np.float32)
    #processed_image = utils.process_image(image, IMG_MEAN)

    with tf.compat.v1.variable_scope("seg_inference"):
        W1_1 = utils.weight_variable([3, 3, 3, 64], name="W1_1")
        b1_1 = utils.bias_variable([64], name="b1_1")
        conv1_1 = utils.conv2d_basic(image, W1_1, b1_1)
        relu1_1 = tf.nn.relu(conv1_1, name="relu1_1")

        W1_2 = utils.weight_variable([3, 3, 64, 64], name="W1_2")
        b1_2 = utils.bias_variable([64], name="b1_2")
        conv1_2 = utils.conv2d_basic(relu1_1, W1_2, b1_2)
        relu1_2 = tf.nn.relu(conv1_2, name="relu1_2")

        ra_1, ra_1_small = utils.RA_unit(relu1_2, relu1_2.shape[1].value, relu1_2.shape[2].value, 16)
        W_s1 = utils.weight_variable([3, 3, 64*(1+16), 64], name="W_s1")
        b_s1 = utils.bias_variable([64], name="b_s1")
        conv_s1 = utils.conv2d_basic(ra_1, W_s1, b_s1)
        relu_s1 = tf.nn.relu(conv_s1, name="relu_s1")
	
        pool1 = utils.max_pool_2x2(relu_s1)

        W2_1 = utils.weight_variable([3, 3, 64, 128], name="W2_1")
        b2_1 = utils.bias_variable([128], name="b2_1")
        conv2_1 = utils.conv2d_basic(pool1, W2_1, b2_1)
        relu2_1 = tf.nn.relu(conv2_1, name="relu2_1")

        W2_2 = utils.weight_variable([3, 3, 128, 128], name="W2_2")
        b2_2 = utils.bias_variable([128], name="b2_2")
        conv2_2 = utils.conv2d_basic(relu2_1, W2_2, b2_2)
        relu2_2 = tf.nn.relu(conv2_2, name="relu2_2")

        ra_2, ra_2_small = utils.RA_unit(relu2_2, relu2_2.shape[1].value, relu2_2.shape[2].value, 16)
        W_s2 = utils.weight_variable([3, 3, 128*(1+16), 128], name="W_s2")
        b_s2 = utils.bias_variable([128], name="b_s2")
        conv_s2 = utils.conv2d_basic(ra_2, W_s2, b_s2)
        relu_s2 = tf.nn.relu(conv_s2, name="relu_s2")
	
        pool2 = utils.max_pool_2x2(relu_s2)

        W3_1 = utils.weight_variable([3, 3, 128, 256], name="W3_1")
        b3_1 = utils.bias_variable([256], name="b3_1")
        conv3_1 = utils.conv2d_basic(pool2, W3_1, b3_1)
        relu3_1 = tf.nn.relu(conv3_1, name="relu3_1")

        W3_2 = utils.weight_variable([3, 3, 256, 256], name="W3_2")
        b3_2 = utils.bias_variable([256], name="b3_2")
        conv3_2 = utils.conv2d_basic(relu3_1, W3_2, b3_2)
        relu3_2 = tf.nn.relu(conv3_2, name="relu3_2")

        W3_3 = utils.weight_variable([3, 3, 256, 256], name="W3_3")
        b3_3 = utils.bias_variable([256], name="b3_3")
        conv3_3 = utils.conv2d_basic(relu3_2, W3_3, b3_3)
        relu3_3 = tf.nn.relu(conv3_3, name="relu3_3")

        ra_3, ra_3_small = utils.RA_unit(relu3_3, relu3_3.shape[1].value, relu3_3.shape[2].value, 16)
        W_s3 = utils.weight_variable([3, 3, 256*(1+16), 256], name="W_s3")
        b_s3 = utils.bias_variable([256], name="b_s3")
        conv_s3 = utils.conv2d_basic(ra_3, W_s3, b_s3)
        relu_s3 = tf.nn.relu(conv_s3, name="relu_s3")

        pool3 = utils.max_pool_2x2(relu_s3)

        W4_1 = utils.weight_variable([3, 3, 256, 512], name="W4_1")
        b4_1 = utils.bias_variable([512], name="b4_1")
        conv4_1 = utils.conv2d_basic(pool3, W4_1, b4_1)
        relu4_1 = tf.nn.relu(conv4_1, name="relu4_1")

        W4_2 = utils.weight_variable([3, 3, 512, 512], name="W4_2")
        b4_2 = utils.bias_variable([512], name="b4_2")
        conv4_2 = utils.conv2d_basic(relu4_1, W4_2, b4_2)
        relu4_2 = tf.nn.relu(conv4_2, name="relu4_2")

        W4_3 = utils.weight_variable([3, 3, 512, 512], name="W4_3")
        b4_3 = utils.bias_variable([512], name="b4_3")
        conv4_3 = utils.conv2d_basic(relu4_2, W4_3, b4_3)
        relu4_3 = tf.nn.relu(conv4_3, name="relu4_3")

        ra_4, ra_4_small = utils.RA_unit(relu4_3, relu4_3.shape[1].value, relu4_3.shape[2].value, 16)
        W_s4 = utils.weight_variable([3, 3, 512*(1+16), 512], name="W_s4")
        b_s4 = utils.bias_variable([512], name="b_s4")
        conv_s4 = utils.conv2d_basic(ra_4, W_s4, b_s4)
        relu_s4 = tf.nn.relu(conv_s4, name="relu_s4")
        
        pool4 = utils.max_pool_2x2(relu_s4)

        W5_1 = utils.weight_variable([3, 3, 512, 512], name="W5_1")
        b5_1 = utils.bias_variable([512], name="b5_1")
        conv5_1 = utils.conv2d_basic(pool4, W5_1, b5_1)
        relu5_1 = tf.nn.relu(conv5_1, name="relu5_1")

        W5_2 = utils.weight_variable([3, 3, 512, 512], name="W5_2")
        b5_2 = utils.bias_variable([512], name="b5_2")
        conv5_2 = utils.conv2d_basic(relu5_1, W5_2, b5_2)
        relu5_2 = tf.nn.relu(conv5_2, name="relu5_2")

        W5_3 = utils.weight_variable([3, 3, 512, 512], name="W5_3")
        b5_3 = utils.bias_variable([512], name="b5_3")
        conv5_3 = utils.conv2d_basic(relu5_2, W5_3, b5_3)
        relu5_3 = tf.nn.relu(conv5_3, name="relu5_3")

        ra_5, ra_5_small = utils.RA_unit(relu5_3, relu5_3.shape[1].value, relu5_3.shape[2].value, 8)
        W_s5 = utils.weight_variable([3, 3, 512*(1+8), 512], name="W_s5")
        b_s5 = utils.bias_variable([512], name="b_s5")
        conv_s5 = utils.conv2d_basic(ra_5, W_s5, b_s5)
        relu_s5 = tf.nn.relu(conv_s5, name="relu_s5")

        pool5 = utils.max_pool_2x2(relu_s5)

        W6 = utils.weight_variable([7, 7, pool4.shape[3].value, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool4, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")

        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")

        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")            #in our case num_of_classess = 2 : road, non-road
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

	# now to upscale to actual image size
        deconv_shape1 = pool3.get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(pool3))
        fuse_1 = tf.add(conv_t1, pool3, name="fuse_1")

        deconv_shape2 = pool2.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool2))
        fuse_2 = tf.add(conv_t2, pool2, name="fuse_2")
        print("fuse_2 shape:")
        print(fuse_2.shape)

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, fuse_2.shape[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=4, stride_y=4)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return annotation_pred, conv_t3				# conv_t3 is the finnal result

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

def next_batch(batch_size, step):
    global batch_offset
    global p
    print("TungNV_batch_offset: ", batch_offset)
    start = batch_offset
    batch_offset += batch_size
    if batch_offset > p.shape[0]:
        print("Shuffle data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Shuffle the data
        perm = np.arange(p.shape[0])
        np.random.shuffle(perm)
        p = p[perm]
        # Start next epoch
        start = 0
        batch_offset = batch_size
    end = batch_offset
    print("train_img start %d end %d"%(start, end))
    return p[start:end]

def check_step_train(path_model):
 	step_str = path_model.split("ckpt-")[1]
 	print("TungNV_step_str: ",step_str)
 	step = int(step_str) + 1
 	print("TungNV_step: ", step)

 	global batch_offset;
 	num_batch = step//p.shape[0]
 	print("TungNV_num_batch: ", num_batch)
 	batch_offset = step - (num_batch*p.shape[0])
 	print("TungNV_batch_offset: ", batch_offset)
 	return step

def load_test_data(path):
	img = io.imread(path)
	print('img.shape', img.shape)

	img = img[:,0:1280,:]
	img = transform.resize(img, (IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH))
	print('TungNV_resize_img.shape', img.shape)
	return img

def predict_image(path_test, keep_probability, image, pred_annotation, prob, sess):
	print("TungNV_path_test = ", path_test)
	test_images = load_test_data(path_test)
	imgs=[]
	imgs.append(test_images)
	images = np.asarray(imgs, np.float32)

	start_time = time.time()
	likelyhood, pred = sess.run([prob, pred_annotation], feed_dict={image:images, keep_probability: 1.0})
	dur = time.time() - start_time
	print("time = ",dur)
	return test_images, likelyhood, pred

def save_image_test(path_test, test_images, pred):
	path, imageName = os.path.split(path_test)
	imageName = imageName[:-4]
	print("TungNV_imageName: ", imageName)

	print("TungNV_imageName_test = ", imageName + ".png")
	path_save_image_test = os.path.join(FLAGS.output_dir, imageName + ".png")
	imageio.imwrite(path_save_image_test, test_images)
	print("TungNV_saved_image_test = ", path_save_image_test)

	for itr in range(pred.shape[0]):
		utils.save_image(pred[itr].astype(np.float32), FLAGS.output_dir, name= imageName + "_pred")
		print("Saved_image_pred: ", imageName + "_pred.png")

def test_image_from_dataset(path, keep_probability, image, pred_annotation, prob, sess):
	p = np.genfromtxt(path, dtype='str')
	#print("TungNV_p = ", p)
	print("TungNV_p.shape[0]: ", p.shape[0])
	for idx in range(0,p.shape[0]):
		path_test = p[idx,0]
		path_test = p[idx,0].replace("..", "/content/drive/My Drive/SingleImageWaterHazardDetectionWithRAU")
		test_images, likelyhood, pred = predict_image(path_test, keep_probability, image, pred_annotation, prob, sess)
		# np.save('./likelyhood/test/likelyhood_%06d'%(idx), likelyhood)
		#print(pred.shape)
		save_image_test(path_test, test_images, pred)

def test_image_from_folder(path, keep_probability, image, pred_annotation, prob, sess):
	listFile = os.listdir(path) # dir is your directory path
	number_files = len(listFile)
	for itr in range(number_files):
		print("TungNV_File: ", listFile[itr])
		path_image_test = os.path.join(path, listFile[itr])
		test_images, likelyhood, pred = predict_image(path_image_test, keep_probability, image, pred_annotation, prob, sess)
		save_image_test(path_image_test, test_images, pred)

def save_video_predict(dir_folder, video_name, fps, reader,  keep_probability, image, pred_annotation, prob, sess):  
    dir_video_out = os.path.join(dir_folder, video_name + '_out.mp4')
    print("TungNV_dir_video_out: ", dir_video_out)
    writer = imageio.get_writer(dir_video_out, fps=fps)

    for num , or_image in enumerate(reader):
        or_height, or_width, or_depth = or_image.shape
        # print("TungNV_or_image.shape: ", or_image.shape)
        if num % 2 == 0:
            print("Predict the frame: ", num)
            or_image = or_image[:,0:1280,:]
            or_image = transform.resize(or_image, (IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH))
            #print("TungNV_or_image.resize.shape: ", or_image.shape)
            imgs=[]
            imgs.append(or_image)
            images = np.asarray(imgs, np.float32)
            likelyhood, predict = sess.run([prob, pred_annotation], feed_dict={image:images, keep_probability: 1.0})

            # print("TungNV_predict.shape: ", predict.shape)
            # predict = predict.reshape(or_height, or_width, 1)
            # predict1 = predict*255
            # predict = np.uint8(np.concatenate((predict, predict, predict1), axis=2))
            # imageOUT = cv2.bitwise_or(or_image, predict)

            # print("TungNV_predict.shape[0]: ", predict.shape[0])
            for itr in range(predict.shape[0]):
                # print("TungNV_itr: ", itr)
                writer.append_data(predict[itr].astype(np.float32))
    writer.close()
    print("Saved_video: ", dir_video_out)

def save_frame(dir_folder, reader, video_length):
    ratio = video_length//10
    print("TungNV_rate: ", ratio)
    for frame , or_image in enumerate(reader):
        or_height, or_width, or_depth = or_image.shape
        # print("TungNV_or_image.shape: ", or_image.shape)
        # pylab.imshow(or_image)
        # pylab.show()
        if frame % ratio == 0:
            dir_image_out = os.path.join(dir_folder, str(frame) + '.png')
            imageio.imwrite(dir_image_out, or_image)
            print("TungNV_Saved_frame: ", frame)

def test_from_video(path, keep_probability, image, pred_annotation, prob, sess):
    reader = imageio.get_reader(path,  'ffmpeg')
    fps = reader.get_meta_data()['fps']
    print("TungNV_fps: ", fps)
    # video_length = reader.count_frames()
    video_length = reader.get_length()
    print("TungNV_video_length: ", video_length)
    dir_folder, video_name = get_name_file(path)

    save_frame(dir_folder, reader, video_length)
    # save_video_predict(dir_folder, video_name, fps, reader, keep_probability, image, pred_annotation, prob, sess)

def get_name_file(path):
    folder_name, name_file = os.path.split(path)
    name_file = name_file[:-4]
    print("TungNV_name_file: ", name_file)
    return folder_name, name_file

def main(argv=None):	
    keep_probability = tf.compat.v1.placeholder(tf.float32, name="keep_probabilty")
    image = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, 3], name="input_image")
    annotation = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, 2], name="annotation")
    pred_annotation, logits = inference(image, keep_probability)			# build the FCN graph

    prob = tf.nn.softmax(logits)

    logits_ = tf.reshape(logits, [1, IMAGE_SIZE_HEIGHT*IMAGE_SIZE_WIDTH, 2])
    annotation_ = tf.reshape(annotation, [1, IMAGE_SIZE_HEIGHT*IMAGE_SIZE_WIDTH, 2])
    loss = utils.focal_loss(logits_, annotation_)

    #loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
     #                                                                     labels=tf.squeeze(annotation, axis=3),
     #                                                                     name="entropy")))
    #tf.summary.image('image', image)
    #tf.summary.image('water_gt', tf.cast(annotation, tf.float32))
    #tf.summary.image('water_pred', tf.expand_dims(tf.cast(pred_annotation, tf.float32), axis=3))

    #tf.summary.scalar('loss', loss)
    #merged = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, graph=tf.get_default_graph())

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config = config)

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    print("TungNV_FLAGS.logs_dir: ", FLAGS.logs_dir)

    ckpt = load_model(saver, sess)

    if FLAGS.mode == "train":
    	print("Loading data")
    	global p;
    	p = load_training_dataset_path()
    	#train_imgs, gt_imgs = load_data(FLAGS.data_dir)
    	#gt_imgs = np.expand_dims(gt_imgs, axis=3)

    	step = check_step_train(ckpt.model_checkpoint_path)

    	print("Start Training...")
    	for itr in xrange(step, MAX_ITERATION):
            print('Step: %d'%(itr))
            p1 = next_batch(1, itr)
            print("p1:\n",p1)
            x_train, y_train, y_train2 = load_data(p1, itr)
            y_train = np.expand_dims(y_train, axis=3)
            y_train2 = np.expand_dims(y_train2, axis=3)
            label_in = np.concatenate((y_train2, y_train), axis=3)
            #x_train, y_train = next_batch(train_imgs, gt_imgs, FLAGS.batch_size)	
            feed_dict = {image: x_train, annotation: label_in, keep_probability: 0.85}
            print("train feed_dict done!")
            start_time = time.time()
            print("Run...")
            sess.run(train_op, feed_dict = feed_dict)
            dur = time.time() - start_time
            print("Time to run : %f"% dur)
            if itr % 1 == 0:
                train_loss = sess.run(loss, feed_dict = feed_dict)
                print('KITTI Step: %d, Train_loss:%g'%(itr, train_loss))
            #if itr % 100 == 0:
            #    summary = sess.run(merged, feed_dict = feed_dict)
            #    summary_writer.add_summary(summary, itr)
            if itr % 5000 == 0:
                print('Save Net Model...')
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            if itr % 5000 == 0 and itr >= 20000:
            	FLAGS.learning_rate = FLAGS.learning_rate / 2

    elif FLAGS.mode == "test":
    	# test_image_from_dataset(PATH_DATASET_TEST, keep_probability, image, pred_annotation, prob, sess)
    	test_image_from_folder(PATH_IMAGE, keep_probability, image, pred_annotation, prob, sess)
      # test_from_video(PATH_VIDEO, keep_probability, image, pred_annotation, prob, sess)


if __name__ == "__main__":
    tf.compat.v1.app.run()
