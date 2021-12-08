#!/usr/bin/env python3
# --*-- coding: utf-8 -*-

import cv2 
import numpy as np
import face_recognition as frec

class FACE():
	def __init__(self):
		self.image_test= frec.load_image_file('./images/test.jpg')
		# self.image_test=cv2.cvtColor(self.image_src, cv2.COLOR_BGR2GRAY)
		self.encode_test=frec.face_encodings(self.image_test)[0]

	def main(self):
		frameWidth = 640
		frameHeight = 480
		cap = cv2.VideoCapture(0)
		cap.set(3, frameWidth)
		cap.set(4, frameHeight)
		cap.set(10,150)
		while cap.isOpened():
			success, self.image_src = cap.read()
			if success:
				import sys
				if self.image_src is None:
					print ("the image read is None............")
					return
				## NEW ##
				# self.enocode_image = frec.face_encodings(self.image_src)[0]
				self.recognize()
				cv2.imshow("Result", self.image_src)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
		cv2.waitKey(3)
		cv2.destroyAllWindows()	
		
	def recognize(self):
		try:
			# self.image_src = cv2.cvtColor(self.image_src, cv2.COLOR_BGR2GRAY)
			floc = frec.face_locations(self.image_src)[0]
			self.encode_face = frec.face_encodings(self.image_src)[0]
			cv2.rectangle(self.image_src,(floc[3],floc[0]),(floc[1],floc[2]),(255,0,255),2)
			self.compare()
		except:
			print("no face recognized")
			
	def compare(self):
		self.name = "Wataru"
		self.result = frec.compare_faces([self.encode_face],self.encode_test)[0]
		self.faceDis = frec.face_distance([self.encode_face],self.encode_test)[0]
		print(self.result,self.faceDis)
		if self.faceDis <.45:
			cv2.putText(self.image_src, f'{self.name}{round(self.faceDis,2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

if __name__ == "__main__":
	fc = FACE()
	fc.main()

