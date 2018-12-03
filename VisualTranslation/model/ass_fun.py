import numpy as np 
import cv2 
import os
from model.config import cfg 
import pdb
import matplotlib.pyplot as plt
import cv2


def generate_batch(N_total, N_each):
	"""
	This file is used to generate index of the training batch.
	
	Arg:
		N_total: 
		N_each: 
	out_put: 
		index_box: the corresponding index
	"""
	num_batch = np.int32(N_total/N_each)
	if N_total%N_each == 0:
		index_box = range(N_total)
	else:
		index_box = np.empty(shape=[N_each*(num_batch+1)],dtype=np.int32)
		index_box[0:N_total] = range(N_total)
		N_rest = N_each*(num_batch+1) - N_total
		index_box[N_total:] = np.random.randint(0,N_total,N_rest)
	return index_box

def generate_batch_bal(labels, N_each):
	N_total = len(labels)
	num_batch = np.int32(N_total/N_each)
	if N_total%N_each == 0:
		index_box = range(N_total)
	else:
		index_box = np.empty(shape=[N_each*(num_batch+1)],dtype=np.int32)
		index_box[0:N_total] = range(N_total)
		N_rest = N_each*(num_batch+1) - N_total

		unique_labels = np.unique(labels, axis = 0)
		N_unique = len(unique_labels)
		num_label = np.zeros([N_unique,])
		for ii in range(N_unique):
			num_label[ii]=np.sum(labels == unique_labels[ii])
		prob_label = np.sum(num_label)/num_label
		prob_label = prob_label/np.sum(prob_label)
		index_rest = np.random.choice(N_unique, size=[N_rest,], p=prob_label)
		for ii in range(N_rest):
			ind = index_rest[ii]
			ind2 = np.where(labels == unique_labels[ind])[0]
			a = np.random.randint(len(ind2))
			index_box[N_total+ii] = ind2[a]
	return index_box

def read_roidb(roidb_path):
	roidb_file = np.load(roidb_path)
	key = roidb_file.keys()[0]
	roidb_temp = roidb_file[key]
	roidb = roidb_temp[()]
	return roidb

def compute_iou(box, proposal):
	"""
	compute the IoU between box with proposal
	Arg:
		box: [x1,y1,x2,y2]
		proposal: N*4 matrix, each line is [p_x1,p_y1,p_x2,p_y2]
	output:
		IoU: N*1 matrix, every IoU[i] means the IoU between
			 box with proposal[i,:]
	"""
	len_proposal = np.shape(proposal)[0]
	IoU = np.empty([len_proposal,1])
	for i in range(len_proposal):
		xA = max(box[0], proposal[i,0])
		yA = max(box[1], proposal[i,1])
		xB = min(box[2], proposal[i,2])
		yB = min(box[3], proposal[i,3])

		if xB<xA or yB<yA:
			IoU[i,0]=0
		else:
			area_I = (xB - xA) * (yB - yA)
			area1 = (box[2] - box[0])*(box[3] - box[1])
			area2 = (proposal[i,2] - proposal[i,0])*(proposal[i,3] - proposal[i,1])
			IoU[i,0] = area_I/float(area1 + area2 - area_I)
	return IoU

def compute_iou_each(box1, box2):
	xA = max(box1[0], box2[0])
	yA = max(box1[1], box2[1])
	xB = min(box1[2], box2[2])
	yB = min(box1[3], box2[3])

	if xB<xA or yB<yA:
		IoU = 0
	else:
		area_I = (xB - xA) * (yB - yA)
		area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
		area2 = (box2[2] - box2[0])*(box2[3] - box2[1])
		IoU = area_I/float(area1 + area2 - area_I)
	return IoU


def im_preprocess(image_path):
	image = cv2.imread(image_path)
	im_orig = image.astype(np.float32, copy=True)
	im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

	im_shape = im_orig.shape
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])

	target_size = cfg.IM_SIZE
	max_size = cfg.IM_MAX_SIZE
	im_scale = float(target_size) / float(im_size_min)

	if np.round(im_scale * im_size_max) > max_size:
		im_scale = float(max_size) / float(im_size_max)
	im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, 
		interpolation=cv2.INTER_LINEAR)
	im_shape_new = np.shape(im)
	im_use = np.zeros([1,im_shape_new[0], im_shape_new[1], im_shape_new[2]])
	im_use[0,:,:,:] = im
	return im_use, im_scale

def get_blob_pred(roidb_use, im_scale, index_sp, N_each_batch, batch_id):
	blob = {}
	sub_box = roidb_use['sub_box_gt']*im_scale
	obj_box = roidb_use['obj_box_gt']*im_scale
	rela = np.int32(roidb_use['rela_gt'])
	index = roidb_use['index_pred']

	index_use = index[batch_id*N_each_batch: (batch_id+1)*N_each_batch]
	sub_box_use = sub_box[index_use,:]
	obj_box_use = obj_box[index_use,:]
	rela_use = rela[index_use]

	blob['sub_box'] = sub_box_use
	blob['obj_box'] = obj_box_use
	blob['rela'] = rela_use
	return blob

def get_blob_rela(roidb_use, im_scale, index_sp, N_each_batch, batch_id):
	blob = {}
	sub_box = roidb_use['sub_box_dete']*im_scale
	obj_box = roidb_use['obj_box_dete']*im_scale
	rela = np.int32(roidb_use['rela_dete'])
	index = roidb_use['index_rela']

	index_use = index[batch_id*N_each_batch: (batch_id+1)*N_each_batch]
	sub_box_use = sub_box[index_use,:]
	obj_box_use = obj_box[index_use,:]
	rela_use = rela[index_use]

	blob['sub_box'] = sub_box_use
	blob['obj_box'] = obj_box_use
	blob['rela'] = rela_use
	return blob

def pred_recall(test_roidb, pred_roidb, N_recall):
	N_right = 0.0
	N_total = 0.0
	N_data = len(test_roidb)
	for i in range(N_data):
		gt_rela = test_roidb[i]['rela_gt']
		if len(gt_rela) == 0:
			continue
		pred_rela = pred_roidb[i]['pred_rela']
		pred_rela_score = pred_roidb[i]['pred_rela_score']
		N_rela = len(gt_rela)
		N_total = N_total + N_rela
		if  N_rela <= N_recall:
			N_right = N_right + np.sum( np.float32(gt_rela == pred_rela) )
		else:
			sort_score = np.sort(pred_rela_score)[::-1]
			thresh = sort_score[N_recall]
			for j in range(N_rela):
				if pred_rela_score[j] >= thresh:
					N_right = N_right + np.float32( gt_rela[j] == pred_rela[j])

	acc = N_right/N_total
	return acc

def rela_recall(test_roidb, pred_roidb, N_recall):
	sublabels = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses", "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel", "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench",     "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog",     "mountain", "horse", "plane", "roof", "skateboard", "traffic light", "bush"    , "phone", "airplane", "sofa", "cup", "sink", "shelf", "box", "van", "hand",     "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer", "pillow", "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can"    , "vase", "refrigerator", "cart", "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe", "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face", "street", "ramp", "suitcase"]
	predlabels = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]
	N_right = 0.0
	N_total = 0.0
	N_data = len(test_roidb)
	num_right = np.zeros([N_data,])
	for i in range(N_data):
		im = cv2.imread(test_roidb[i]['image'])
		rela_gt = test_roidb[i]['rela_gt']
		if len(rela_gt) == 0:
			continue
		sub_gt = test_roidb[i]['sub_gt']
		obj_gt = test_roidb[i]['obj_gt']
		sub_box_gt = test_roidb[i]['sub_box_gt']
		obj_box_gt = test_roidb[i]['obj_box_gt']

		pred_rela = pred_roidb[i]['pred_rela']
		pred_rela_score = pred_roidb[i]['pred_rela_score']
		sub_dete = pred_roidb[i]['sub_dete']
		obj_dete = pred_roidb[i]['obj_dete']
		sub_box_dete = pred_roidb[i]['sub_box_dete']
		obj_box_dete = pred_roidb[i]['obj_box_dete']

		N_rela = len(rela_gt)
		N_total = N_total + N_rela

		N_pred = len(pred_rela)
		
		sort_score = np.sort(pred_rela_score)[::-1]
		if N_recall >= N_pred:
			thresh = -1
		else:
			thresh = sort_score[N_recall]

		detected_gt = np.zeros([N_rela,])
		for j in range(N_pred):
			if pred_rela_score[j] <= thresh:
				continue

			im2 = im.copy()
			sub_box = sub_box_dete[j].astype(int)
			obj_box = obj_box_dete[j].astype(int)
			cv2.rectangle(im2, (sub_box[0], sub_box[1]), (sub_box[2], sub_box[3]), (255,0,0),3)
			cv2.rectangle(im2, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]), (0,0,255),3)
			cv2.putText(im2, "Sub: " + sublabels[int(sub_dete[j])], (sub_box[0], sub_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
			cv2.putText(im2, "Obj: " + sublabels[int(obj_dete[j])], (obj_box[0], obj_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
			cv2.putText(im2, "Pred: " + predlabels[int(pred_rela[j])] + " Score: " + str(pred_rela_score[j]), (obj_box[0], obj_box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
			path, file = os.path.split(test_roidb[i]['image'])
			path = path[:-14] + "results/" + file[:-4] + "spo" + str(j) + ".png"
			
			# uncomment the following line to save the results in /dataset/VRD/sg_dataset/results
			# create the results folder first
			# cv2.imwrite(path, im2)


			for k in range(N_rela):
				if detected_gt[k] == 1:
					continue
				if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]) and (rela_gt[k] == pred_rela[j]):
					s_iou = compute_iou_each(sub_box_dete[j], sub_box_gt[k])
					o_iou = compute_iou_each(obj_box_dete[j], obj_box_gt[k])
					if ( s_iou >= 0.5 ) and ( o_iou >=0.5 ):
						detected_gt[k] = 1
						N_right = N_right + 1
						num_right[i] = num_right[i] + 1

	acc = N_right/N_total
	print(N_right)
	print(N_total)
	return acc, num_right

def phrase_recall(test_roidb, pred_roidb, N_recall):
	N_right = 0.0
	N_total = 0.0
	N_data = len(test_roidb)
	num_right = np.zeros([N_data,])
	for i in range(N_data):
		rela_gt = test_roidb[i]['rela_gt']
		if len(rela_gt) == 0:
			continue

		N_rela = len(rela_gt)
		sub_gt = test_roidb[i]['sub_gt']
		obj_gt = test_roidb[i]['obj_gt']
		sub_box_gt = test_roidb[i]['sub_box_gt']
		obj_box_gt = test_roidb[i]['obj_box_gt']
		phrase_gt = generate_phrase_box(sub_box_gt, obj_box_gt)


		pred_rela = pred_roidb[i]['pred_rela']
		pred_rela_score = pred_roidb[i]['pred_rela_score']
		sub_dete = pred_roidb[i]['sub_dete']
		obj_dete = pred_roidb[i]['obj_dete']
		sub_box_dete = pred_roidb[i]['sub_box_dete']
		obj_box_dete = pred_roidb[i]['obj_box_dete']
		phrase_dete = generate_phrase_box(sub_box_dete, obj_box_dete)
		N_pred = len(pred_rela)

		N_total = N_total + N_rela

		sort_score = np.sort(pred_rela_score)[::-1]
		if N_recall >= N_pred:
			thresh = -1
		else:
			thresh = sort_score[N_recall]

		detected_gt = np.zeros([N_rela,])
		for j in range(N_pred):
			if pred_rela_score[j] <= thresh:
				continue
			
			for k in range(N_rela):
				if detected_gt[k] == 1:
					continue
				if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]) and (rela_gt[k] == pred_rela[j]):
					iou = compute_iou_each(phrase_dete[j], phrase_gt[k])
					if  iou >= 0.5 :
						detected_gt[k] = 1
						N_right = N_right + 1
						num_right[i] = num_right[i] + 1

	acc = N_right/N_total
	print(N_right)
	print(N_total)
	return acc, num_right

def print_pred_res(test_roidb, pred_roidb, res_name, N_rela):
	N_pred_rela = np.zeros([N_rela,])
	N_right_rela = np.zeros([N_rela,])
	N_gt_rela = np.zeros([N_rela,])
	N_acc = np.zeros([N_rela,])
	N_pred_other = np.zeros([N_rela,N_rela])

	N_pred = len(pred_roidb)
	for i in range(N_pred):
		pred_rela = np.int32(pred_roidb[i]['pred_rela'])
		gt_rela = np.int32(test_roidb[i]['rela_gt'])
		for k in range(len(gt_rela)):
			N_pred_rela[pred_rela[k]] += 1
			N_gt_rela[gt_rela[k]] += 1
			N_pred_other[gt_rela[k]][pred_rela[k]] += 1
			if pred_rela[k]==gt_rela[k]:
				N_right_rela[gt_rela[k]] += 1

	for k in range(N_rela):
		if N_gt_rela[k] == 0:
			N_acc[k] = -1
			continue
		N_acc[k] = (N_right_rela[k]+0.0)/N_gt_rela[k]

	text_file = open(res_name,"aw")
	for k in range(N_rela):
		text_file.write('k: {0}, acc: {1}\n'.format(k, N_acc[k]))
	for k in range(N_rela):
		text_file.write('k: {0}, num of pred: {1}\n'.format(k, N_pred_rela[k]))
	for k in range(N_rela):
		text_file.write('k: {0}, others: {1}\n'.format(k,np.where(N_pred_other[k]>0)[0]))
	text_file.close()




def generate_phrase_box(sbox, obox):
	N_box = len(sbox)
	phrase = np.zeros([N_box,4])
	for i in range(N_box):
		phrase[i,0] = min(sbox[i,0], obox[i,0])
		phrase[i,1] = min(sbox[i,1], obox[i,1])
		phrase[i,2] = max(sbox[i,2], obox[i,2])
		phrase[i,3] = max(sbox[i,2], obox[i,3])
	return phrase

def extract_detected_box(dete_box):
	N_cls = len(dete_box['pred_boxes'])
	pred_box = dete_box['pred_boxes']
	detected_cls = np.zeros([1000,])
	detected_box = np.zeros([1000,4])
	detected_score = np.zeros([1000,])
	t = 0
	for i in range(N_cls):
		l = len(pred_box[i])
		if l == 0:
			continue
		box_temp = pred_box[i]
		detected_box[t:t+l] = box_temp[:,0:4]
		detected_cls[t:t+l] = i
		detected_score[t:t+l] = box_temp[:,4]
		t = t + l
	detected_box = detected_box[0:t]
	detected_cls = detected_cls[0:t]
	detected_score = detected_score[0:t]
	return detected_box, detected_cls, detected_score

def extract_roidb_box(roidb):
	sbox = roidb['sub_box_gt']
	obox = roidb['obj_box_gt']
	roidb_box = np.concatenate( (roidb['sub_box_gt'], roidb['obj_box_gt']), axis = 0 )
	roidb_cls = np.concatenate( (roidb['sub_gt'], roidb['obj_gt']), axis = 0)
	roidb_rela = roidb['rela_gt']
	N_rela = len(roidb_rela)
	unique_boxes, unique_inds = np.unique(roidb_box, axis=0, return_index = True)
	unique_cls = roidb_cls[unique_inds]
	rela_box_index = np.zeros( [N_rela,2] )
	for i in range(N_rela):
		rela_box_index[i,0] = np.where(unique_boxes == sbox[i])[0][0]
		rela_box_index[i,1] = np.where(unique_boxes == obox[i])[0][0]
	return unique_cls, unique_boxes, roidb_rela, rela_box_index

def generate_au_box(unique_boxes, detected_box, iou_l):
	N_unique = len(unique_boxes)
	au_box = []
	for i in range(N_unique):
		box_temp = unique_boxes[i]
		iou = compute_iou(box_temp, detected_box)
		index_temp = np.where(iou > iou_l)[0]
		box_use = detected_box[index_temp]
		box_use = np.vstack( (box_use, box_temp ) )
		au_box.append(box_use)
	return au_box

def generate_rela_info(au_box, index, N_each_pair):
	s_id = np.int32(index[0])
	o_id = np.int32(index[1])
	sbox = au_box[s_id]
	obox = au_box[o_id]
	N_s = len(sbox)
	N_o = len(obox)
	sa = np.random.randint(0, N_s, [N_each_pair,])
	oa = np.random.randint(0, N_o, [N_each_pair,])
	sbox_use = sbox[sa]
	obox_use = obox[oa]
	return sbox_use, obox_use

def generate_train_rela_roidb(roidb, dete_box, iou_l, N_each_batch, N_each_pair):
	unique_cls, unique_boxes, roidb_rela, rela_box_index = extract_roidb_box(roidb)
	detected_box, detected_cls, detected_score = extract_detected_box(dete_box)
	au_box = generate_au_box(unique_boxes, detected_box, iou_l)

	N_rela = len(roidb_rela)
	for i in range(N_rela):
		sbox, obox = generate_rela_info(au_box, rela_box_index[i], N_each_pair)
		s_id = np.int32(rela_box_index[i,0])
		o_id = np.int32(rela_box_index[i,1])
		sb = np.zeros([N_each_pair,]) + unique_cls[s_id]
		ob = np.zeros([N_each_pair,]) + unique_cls[o_id]
		rela = np.zeros([N_each_pair,]) + roidb_rela[i]
		if i == 0:
			sub_box_dete = sbox
			obj_box_dete = obox
			sub_dete = sb
			obj_dete = ob 
			rela_dete = rela 
		else:
			sub_box_dete = np.vstack( (sub_box_dete, sbox) )
			obj_box_dete = np.vstack( (obj_box_dete, obox) )
			sub_dete = np.concatenate( (sub_dete, sb), axis = 0 )
			obj_dete = np.concatenate( (obj_dete, ob), axis = 0 )
			rela_dete = np.concatenate( (rela_dete, rela), axis = 0 )

	index_rela = generate_batch(len(rela_dete), N_each_batch)

	roidb_temp = {'image': roidb['image'], 'au_box': au_box,'rela_box_index': rela_box_index,
				  'unique_cls': unique_cls, 'unique_box': unique_boxes, 'rela_gt': roidb_rela,
				  'sub_box_dete': sub_box_dete, 'obj_box_dete': obj_box_dete, 'sub_dete': sub_dete,
				  'obj_dete': obj_dete, 'rela_dete': rela_dete, 'index_rela': index_rela}
	return roidb_temp

def generate_test_rela_roidb(roidb, dete_box, N_each_batch):
	detected_box, detected_cls, detected_score = extract_detected_box(dete_box)
	N_dete = len(detected_box)
	sub_box_dete = np.zeros([N_dete*(N_dete-1),4])
	obj_box_dete = np.zeros([N_dete*(N_dete-1),4])
	sub_dete = np.zeros([N_dete*(N_dete-1),])
	obj_dete = np.zeros([N_dete*(N_dete-1),])
	sub_score = np.zeros([N_dete*(N_dete-1),])
	obj_score = np.zeros([N_dete*(N_dete-1),])

	t = 0
	for i in range(N_dete):
		for j in range(N_dete):
			if j == i:
				continue
			sub_box_dete[t] = detected_box[i]
			obj_box_dete[t] = detected_box[j]
			sub_dete[t] = detected_cls[i]
			obj_dete[t] = detected_cls[j]
			sub_score[t] = detected_score[i]
			obj_score[t] = detected_score[j]
			t = t+1
	sub_box_dete = sub_box_dete[0:t]
	obj_box_dete = obj_box_dete[0:t]
	sub_dete = sub_dete[0:t]
	obj_dete = obj_dete[0:t]
	sub_score = sub_score[0:t]
	obj_score = obj_score[0:t]
	index_rela = generate_batch(len(sub_box_dete), N_each_batch)
	rela_dete = np.zeros([len(sub_box_dete),])

	roidb_temp = {'image': roidb['image'], 'rela_gt': roidb['rela_gt'], 'sub_gt': roidb['sub_gt'], 'obj_gt': roidb['obj_gt'],
				  'sub_box_dete': sub_box_dete, 'obj_box_dete': obj_box_dete, 'sub_dete': sub_dete, 'obj_dete': obj_dete,
				  'detected_box': detected_box, 'detected_cls': detected_cls, 'sub_box_gt': roidb['sub_box_gt'],
				  'obj_box_gt': roidb['obj_box_gt'], 'sub_score': sub_score, 'obj_score': obj_score, 'index_rela': index_rela,
				  'rela_dete': rela_dete}

	return roidb_temp
