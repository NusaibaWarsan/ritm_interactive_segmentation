import cv2
import numpy as np

class ObjSegmentation:

	def __init__(self, image, gt_points, anno_points, ritm_mask):
		self.image = cv2.imread(image)
		self.gt_mask = self.get_mask(gt_points)
		self.agent_mask = self.get_mask(anno_points)
		self.ritm_mask = cv2.imread(ritm_mask)
		self.grabcut_mask = self.grabcut(cv2.imread(ritm_mask, cv2.IMREAD_GRAYSCALE))

		self.values = (
			("Definite Background", cv2.GC_BGD),
			("Probable Background", cv2.GC_PR_BGD),
			("Definite Foreground", cv2.GC_FGD),
			("Probable Foreground", cv2.GC_PR_FGD),
		)


	def get_mask(self, points):
		'''
		Create a binary segmentation mask using polygon annotation coordinates
		'''
		np_points = []
		for point in points:
			np_points.append([point['x'], point['y']])
		area = np.array(np_points)
		filled = cv2.fillPoly(
			np.zeros_like(self.image), pts=np.int32([area]), color=(255,255,255))
		return filled

	def get_iou(self, gt, anno):
		intersection = np.logical_and(gt, anno)
		union = np.logical_or(gt, anno)
		iou = np.sum(intersection)/np.sum(union)
		return iou

	def grabcut(self, mask):
		mask[mask > 0] = cv2.GC_PR_FGD
		mask[mask == 0] = cv2.GC_BGD
		fgModel = np.zeros((1, 65), dtype="float")
		bgModel = np.zeros((1, 65), dtype="float")
		(new_mask, bgModel, fgModel) = cv2.grabCut(
			self.image, mask, None, bgModel,fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
		outputMask = np.where((new_mask == cv2.GC_BGD) | (new_mask == cv2.GC_PR_BGD), 0, 1)
		outputMask = (outputMask * 255).astype("uint8")
		outputMask=cv2.cvtColor(outputMask, cv2.COLOR_BGR2RGB)
		return outputMask

	def clahe(self):	
		lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
		clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
		lab[...,0] = clahe.apply(lab[...,0])
		bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
		return bgr

	def save_enhanced_image(self):
		enhanced = self.clahe()
		cv2.imwrite('enhanced.png', enhanced)


