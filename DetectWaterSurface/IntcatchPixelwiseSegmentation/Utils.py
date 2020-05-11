import cv2
import numpy as np
import scipy
import glob
import matplotlib.pyplot as plt
from Line_of_horizont_fitting import Line_of_horizont_fitting
import imageio
from time import time, sleep
import pylab
import os

class Utils:
    @staticmethod
    def median_accuracy_line_of_horizont(x, y, model, inp_w, inp_h, steps=1, visualization=False):
        avg_distance=[]
        max_distance=[]
        
        line_of_horizont=Line_of_horizont_fitting()
        for label, img in zip(y, x):

            image = np.uint8(255 * img)
            
            label_med=line_of_horizont.median_blur(label,5)
            label_med=line_of_horizont.get_binary_image(label_med, 0.5)
            fit_line= line_of_horizont.horizont_line_from_binary_image(label_med)

            label=line_of_horizont.get_binary_image(label, 0.5)
            height,width=label.shape
            label_line = np.zeros([height,width], dtype = "uint8")
            cv2.line(label_line, (int(fit_line[2]-fit_line[0]*width), 
                                             int(fit_line[3]-fit_line[1]*width)), 
                     (int(fit_line[2]+fit_line[0]*width), 
                      int(fit_line[3]+fit_line[1]*width)), (255, 255, 255), 1)

            image_pred = line_of_horizont.resize_image(img, inp_w, inp_h)
            pred=line_of_horizont.predict_segmentation(image_pred, model)
            pred=line_of_horizont.get_binary_image(pred, 0.5)
            pred=line_of_horizont.resize_image(pred, width, height)

            fit_line, predict=line_of_horizont.horizont_line_pipeline(image, model, inp_w, inp_h, steps, 5)

            pred_line = np.zeros([height,width], dtype = "uint8")
            cv2.line(pred_line, (int(fit_line[2]-fit_line[0]*width), 
                                             int(fit_line[3]-fit_line[1]*width)), 
                     (int(fit_line[2]+fit_line[0]*width), 
                      int(fit_line[3]+fit_line[1]*width)), (255, 255, 255), 1)

            distance=[]
            for j in range (width):
                for i in range (height):
                    if(label_line[i,j]==255):
                        y1=i
                    if(pred_line[i,j]==255):
                        y2=i
                distance.append(abs(y1-y2))

            avg_y= int((y1+y2)/2)
            
            avg_distance.append(np.mean(distance)/width)
            max_distance.append(max(distance))
            
            if(visualization):
                print("avg_distance: ", (np.mean(distance)/width)," - max_distance: ", (max(distance)))
                plt.imshow(label_line)
                plt.show()
                plt.imshow(pred_line)
                plt.show()
        
        return avg_distance, max_distance
    
    @staticmethod
    def accuracy_on_line_of_horizont_area(x, y, model, inp_w, inp_h, steps=1, visualization=False):
        recall_list=[]
        precision_list=[]
        specificity_list=[]
        accuracy_list=[]
        f1score_list=[]

        line_of_horizont=Line_of_horizont_fitting()
        for label, img in zip(y, x):

            image = np.uint8(255 * img)
            label=line_of_horizont.get_binary_image(label, 0.5)
            height,width=label.shape

            image_pred = line_of_horizont.resize_image(img, inp_w, inp_h)
            pred=line_of_horizont.predict_segmentation(image_pred, model)
            pred=line_of_horizont.get_binary_image(pred, 0.5)
            pred=line_of_horizont.resize_image(pred, width, height)

            fit_line, predict=line_of_horizont.horizont_line_pipeline(image, model, inp_w, inp_h, steps)
            #fit_line, predict=line_of_horizont.horizont_line_from_binary_image(image, model, inp_w, inp_h, steps)
            line_annotation_image = np.zeros([height,width], dtype = "uint8")
            cv2.line(line_annotation_image, (int(fit_line[2]-fit_line[0]*width), 
                                             int(fit_line[3]-fit_line[1]*width)), 
                     (int(fit_line[2]+fit_line[0]*width), 
                      int(fit_line[3]+fit_line[1]*width)), (255, 255, 255), 1)

            for i in range (height):
                if(label[i,0]==1):
                    y1=i
                    break

            for i in range (height):
                if(label[i,width-1]==1):
                    y2=i
                    break

            avg_y= int((y1+y2)/2)

            annotation_image = label[avg_y-100:avg_y+100, 0:width]
            pred_image = pred[avg_y-100:avg_y+100, 0:width]

            label=annotation_image
            pred=pred_image

            True_neg=len(np.where((label==0)&(pred==0))[0])
            False_neg=len(np.where((label==1)&(pred==0))[0])
            True_pos=len(np.where((label==1)&(pred==1))[0])
            False_pos=len(np.where((label==0)&(pred==1))[0])
            precision=True_pos/(True_pos+False_pos)
            recall=True_pos/(True_pos+False_neg)
            specificity=1-(True_neg/(True_neg+False_pos))
            accuracy=(True_pos+True_neg)/(True_pos+True_neg+False_pos+False_neg)
            f1score=2*((precision*recall)/(precision+recall))

            recall_list.append(recall)
            precision_list.append(precision)
            specificity_list.append(specificity)
            accuracy_list.append(accuracy)
            f1score_list.append(f1score)

            if(visualization):
                print("Recall: ", recall," - Precision: ", precision, " - Specificity: ", specificity, " - Accuracy: ", 
                      accuracy, " - F1score: ", f1score)
                plt.imshow(label)
                plt.show()
                plt.imshow(pred)
                plt.show()

        return recall_list, precision_list, specificity_list, accuracy_list, f1score_list
    
    @staticmethod
    def accuracy_on_images(x, y, model, inp_w, inp_h, steps=1, visualization=False):
        recall_list=[]
        precision_list=[]
        specificity_list=[]
        accuracy_list=[]
        f1score_list=[]
        
        line_of_horizont=Line_of_horizont_fitting()
        for label, img in zip(y, x):

            label=line_of_horizont.get_binary_image(label, 0.5)
            height,width=label.shape
            image_pred = line_of_horizont.resize_image(img, 160, 160)
            pred=line_of_horizont.predict_segmentation(image_pred, model)
            pred=line_of_horizont.get_binary_image(pred, 0.5)
            pred=line_of_horizont.resize_image(pred, width, height)

            True_neg=len(np.where((label==0)&(pred==0))[0])
            False_neg=len(np.where((label==1)&(pred==0))[0])
            True_pos=len(np.where((label==1)&(pred==1))[0])
            False_pos=len(np.where((label==0)&(pred==1))[0])
            precision=True_pos/(True_pos+False_pos)
            recall=True_pos/(True_pos+False_neg)
            specificity=1-(True_neg/(True_neg+False_pos))
            accuracy=(True_pos+True_neg)/(True_pos+True_neg+False_pos+False_neg)
            f1score=2*((precision*recall)/(precision+recall))
            
            recall_list.append(recall)
            precision_list.append(precision)
            specificity_list.append(specificity)
            accuracy_list.append(accuracy)
            f1score_list.append(f1score)
            
            if(visualization):
                print("Recall: ", recall," - Precision: ", precision, " - Specificity: ", specificity, " - Accuracy: ", 
                      accuracy, " - F1score: ", f1score)
                plt.imshow(label)
                plt.show()
                plt.imshow(pred)
                plt.show()
            
        return recall_list, precision_list, specificity_list, accuracy_list, f1score_list
            
    @staticmethod
    def test_speed_from_video(filename, model, inp_w, inp_h, n_iteration, steps=1):
        lineofhorizont = Line_of_horizont_fitting()
        reader = imageio.get_reader(filename,  'ffmpeg')
        fps = reader.get_meta_data()['fps']
        n_steps=0
        frame_to_discard = 10
        now=time()
        for i in range(n_iteration):
            if i == frame_to_discard:
                start_time=now
            if i > frame_to_discard:
                #n_steps+=1
                elapsed_time = now - start_time
                #print(n_steps, elapsed_time)
            or_image=reader.get_data(i)
            or_height, or_width, or_depth = or_image.shape

            fit_line, predict=lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, steps)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_height), int(fit_line[3]-fit_line[1]*or_width)), 
                              (int(fit_line[2]+fit_line[0]*or_height), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 5)
            now=time()
            
        reader.close()
        return ((i-frame_to_discard))/elapsed_time
    
    @staticmethod
    def test_speed_from_video_v2(reader, model, inp_w, inp_h, n_iteration, steps=1):
        n_steps=0
        frame_to_discard = 10
        now=time()
        lineofhorizont = Line_of_horizont_fitting()
        for i in range(n_iteration):
            if i == frame_to_discard:
                start_time=now
            if i > frame_to_discard:
                #n_steps+=1
                elapsed_time = now - start_time
                #print(n_steps, elapsed_time)
            or_image=reader.get_data(i)
            or_height, or_width, or_depth = or_image.shape
            
            fit_line, predict=lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, steps)
   
            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_height), int(fit_line[3]-fit_line[1]*or_width)), 
                              (int(fit_line[2]+fit_line[0]*or_height), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 5)
            now=time()

        return ((i-frame_to_discard))/elapsed_time
 
    @staticmethod
    def test_from_video(filename, model, inp_w, inp_h, n_iteration, steps=1):
        lineofhorizont = Line_of_horizont_fitting()
        reader = imageio.get_reader(filename,  'ffmpeg')
        fps = reader.get_meta_data()['fps']
        now=time()
        start_time=now
        for i in range(n_iteration):
            elapsed_time = now - start_time
            print(i, elapsed_time)
            or_image=reader.get_data(i)
            # plt.imshow(or_image)
            # plt.show()
            or_height, or_width, or_depth = or_image.shape
            print("TungNV_or_image.shape: ", or_image.shape)

            fit_line, predict = lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, steps)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_height), int(fit_line[3]-fit_line[1]*or_width)), 
                              (int(fit_line[2]+fit_line[0]*or_height), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 5)
            now=time()
            plt.imshow(imageOUT)
            plt.show()
        reader.close()

    @staticmethod
    def test_from_video_TungNV(dir_video, model, inp_w, inp_h, n_iteration, steps=1):
        lineofhorizont = Line_of_horizont_fitting()
        reader = imageio.get_reader(dir_video,  'ffmpeg')
        fps = reader.get_meta_data()['fps']
        print("TungNV_fps: ", fps)
        video_length = reader.count_frames()
        print("TungNV_video_length: ", video_length)

        dir_folder, video_name = Utils.get_name_file(dir_video)
        dir_video_out = os.path.join(dir_folder, video_name + '_out.mp4')
        print("TungNV_dir_video_out: ", dir_video_out)
        writer = imageio.get_writer(dir_video_out, fps=fps)

        now=time()
        start_time=now
        # print("TungNV_reader.size: ", reader.size())
        for num , or_image in enumerate(reader):
            elapsed_time = now - start_time
            or_height, or_width, or_depth = or_image.shape
            # print("TungNV_or_image.shape: ", or_image.shape)
            # pylab.imshow(or_image)
            # pylab.show()

            # if num == 5700:
            #     dir_image_out = os.path.join(dir_folder, str(num) + '.png')
            #     imageio.imwrite(dir_image_out, or_image[:, :, 0])
            #     print("TungNV_Save_frame: ", num)
            #     break

            if num % 2 == 0:
                print("Predict the frame: ", num)
                fit_line, predict = lineofhorizont.horizont_line_pipeline_TungNV(or_image, model, inp_w, inp_h, steps)
                #print("TungNV_predict.all: ", predict.all())
                if predict is None:
                    imageOUT = or_image
                else:
                    predict = predict.reshape(or_height, or_width, 1)
                    predict1 = predict*255
                    predict = np.uint8(np.concatenate((predict, predict, predict1), axis=2))
                    imageOUT = cv2.bitwise_or(or_image, predict)
                    #imageOUT = cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_height), int(fit_line[3]-fit_line[1]*or_width)), (int(fit_line[2]+fit_line[0]*or_height), int(fit_line[3]+fit_line[1]*or_width)), (255, 0, 255), 5)
                
                now=time()
                writer.append_data(imageOUT)
        writer.close()
        print("Saved_video: ", dir_video_out)

    @staticmethod
    def test_from_folder(path, dir_output, model, inp_w, inp_h, steps=1):
        lineofhorizont = Line_of_horizont_fitting()
        # path_images=glob.glob(path)
        path_images = glob.glob(os.path.join(path, '*.png'))
        #print("TungNV_path_images: ", path_images)

        images=[]
        now=time()
        start_time=now

        for path_img in path_images:
            print("TungNV_path_img: ", path_img)
            elapsed_time = now - start_time
            or_image=cv2.imread(path_img)
            or_image=cv2.cvtColor(or_image, cv2.COLOR_BGR2RGB)
            or_height, or_width, or_depth = or_image.shape
            print("TungNV_or_image.shape: ", or_image.shape)

            fit_line, predict, img_inp_or, pred_inp_or=lineofhorizont.horizont_line_pipeline_verbose(or_image, model, inp_w, inp_h, steps)
            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            
            #(x0-m*vx[0], y0-m*vy[0]), (x0+m*vx[0], y0+m*vy[0])
            
            W = or_width 
            H = or_height
            x0 = (int(fit_line[2]-(W*fit_line[0])))
            x1 = (int(fit_line[2]+(W*fit_line[0])))
            y0 = (int(fit_line[3]-(H*fit_line[1])))
            y1 = (int(fit_line[3]+(H*fit_line[1])))
            imageOUT=cv2.line(imageOUT, (x0,y0), (x1,y1), (255, 0, 255), 5)
            now=time()

            #print("TungNV_imageOUT: ", imageOUT)
            #print("TungNV_predict: ", predict)
            #print("TungNV_img_inp_or: ", img_inp_or)
            #print("TungNV_pred_inp_or: ", pred_inp_or)
            dir_folder, imageName = Utils.get_name_file(path_img)
            dir_image = os.path.join(dir_output, imageName + ".png")
            Utils.save_image_test(dir_image, or_image)
            dir_image_out = os.path.join(dir_output, imageName + "_OUT.png")
            Utils.save_image_test(dir_image_out, imageOUT)
            dir_image_pred = os.path.join(dir_output, imageName + "_pred.png")
            Utils.save_image_test(dir_image_pred, predict)

            # yield path_img, imageOUT, predict, img_inp_or, pred_inp_or

    @staticmethod
    def create_folder (dirName):
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:
            print("Directory " , dirName ,  " already exists")
        return dirName

    @staticmethod
    def get_name_file(path):
        folder_name, name_file = os.path.split(path)
        name_file = name_file[:-4]
        print("TungNV_name_file: ", name_file)
        return folder_name, name_file

    @staticmethod
    def save_image_test(dir_save, images):
        cv2.imwrite(dir_save, images)
        print("TungNV_save_image_test: ", dir_save)
