import scipy.io
import scipy.misc
from torch.autograd import Variable
import argparse
from utils.utils import *
from model import Net
from common import * 
from model_HLFSR_ASR import HLFSR_ASR
import time
import matplotlib.pyplot as plt

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='cuda:0')
	parser.add_argument("--angRes_in", type=int, default=2, help="input angular resolution")
	parser.add_argument("--angRes_out", type=int, default=7, help="output angular resolution")
	parser.add_argument("--model_name", type=str, default='HFLSR_ASR_Lytro_2x2-7x7')
	parser.add_argument("--testset_dir", type=str, default='../Data/TestData_Lytro_2x2-7x7/')
	parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
	parser.add_argument("--patchsize", type=int, default=128, help="LFs are cropped into patches to save GPU memory")
	parser.add_argument('--save_path', type=str, default='./Results/')

	parser.add_argument("--n_groups", type=int, default=10, help="number of HLFSR-Groups")
	parser.add_argument("--n_blocks", type=int, default=15, help="number of HLFSR-Blocks")
	parser.add_argument("--channels", type=int, default=64, help="number of channels")

	
	return parser.parse_args()

def test(cfg):

	test_Names, test_loaders, length_of_tests = MultiTestSetDataLoader(cfg)
	net = HLFSR_ASR(angRes_in=cfg.angRes_in, angRes_out=cfg.angRes_out, n_blocks=cfg.n_blocks,
				   channels=cfg.channels)
	net.to(cfg.device)

	total_params = sum(p.numel() for p in net.parameters())
	print("Total Params: {:.2f}".format(total_params)) 

	model = torch.load('./log/' + cfg.model_name + '.pth.tar', map_location={'cuda:1': cfg.device})
	net.load_state_dict(model['state_dict'])

	with torch.no_grad():
		psnr_testset = []
		ssim_testset = []
		for index, test_name in enumerate(test_Names):
			test_loader = test_loaders[index]
			outLF, psnr_epoch_test, ssim_epoch_test = valid(test_loader, test_name, net)
			# outLF, psnr_epoch_test, ssim_epoch_test = valid_crop(test_loader, test_name, net,cfg.angRes_in, cfg.angRes_out)
			psnr_testset.append(psnr_epoch_test)
			ssim_testset.append(ssim_epoch_test)
			print('Dataset----%10s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
			txtfile = open(cfg.save_path + cfg.model_name + '_metrics.txt', 'a')
			txtfile.write('Dataset----%10s,\t PSNR---%f,\t SSIM---%f\n' % (test_name, psnr_epoch_test, ssim_epoch_test))
			txtfile.close()
			pass
		pass


def valid(test_loader, test_name, net):
	psnr_iter_test = []
	ssim_iter_test = []
	for idx_iter, (data, label) in (enumerate(test_loader)):
		data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
		label = label.squeeze().to(cfg.device)
		if cfg.crop == False:
			with torch.no_grad():
				outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
				outLF = outLF.squeeze()
		else:
			patchsize = cfg.patchsize
			stride = patchsize // 2
			uh, vw = data.shape
			h0, w0 = uh // cfg.angRes_in, vw // cfg.angRes_in
			subLFin = LFdivide(data, cfg.angRes_in, patchsize, stride)  # numU, numV, h*angRes, w*angRes
			numU, numV, H, W = subLFin.shape
			subLFout = torch.zeros(numU, numV, cfg.angRes_out * patchsize, cfg.angRes_out * patchsize)

			for u in range(numU):
				for v in range(numV):
					tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
					with torch.no_grad():
						torch.cuda.empty_cache()
						out = net(tmp.to(cfg.device))
						subLFout[u, v, :, :] = out.squeeze()

			outLF = LFintegrate(subLFout, cfg.angRes_out, patchsize, stride, h0, w0)

		psnr, ssim = cal_metrics_RE(label, outLF, cfg.angRes_in, cfg.angRes_out)

		psnr_iter_test.append(psnr)
		ssim_iter_test.append(ssim)

		if not (os.path.exists(cfg.save_path + '/' + test_name)):
			os.makedirs(cfg.save_path + '/' + test_name)
		scipy.io.savemat(cfg.save_path + '/' + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
						 {'LF': outLF.cpu().numpy()})
		pass


	psnr_epoch_test = float(np.array(psnr_iter_test).mean())
	ssim_epoch_test = float(np.array(ssim_iter_test).mean())

	return outLF, psnr_epoch_test, ssim_epoch_test


	psnr_iter_test = []
	ssim_iter_test = []
	time_total =0

	for idx_iter, (data, label) in (enumerate(test_loader)):
		data = data.to(cfg.device)  # numU, numV, h*angRes, w*angRes
		label = label.squeeze().to(cfg.device)

		

		if cfg.crop == False:
			with torch.no_grad():
				outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
				outLF = outLF.squeeze()
		else:
			data = SAI2MacPI(data,angRes_in)
			b, c, h, w = data.size()

			# img_tmp = data.squeeze()
			# imgplot = plt.imshow(img_tmp.cpu().numpy())
			# plt.show()

			scale  = angRes_out
			h_half, w_half = h // 2, w // 2
			h_size, w_size = h_half , w_half 

			lr_list = [
				data[:, :, 0:h_size, 0:w_size],
				data[:, :, 0:h_size, (w - w_size):w],
				data[:, :, (h - h_size):h, 0:w_size],
				data[:, :, (h - h_size):h, (w - w_size):w]]


			sr_list = []
			n_GPUs = 1
			for i in range(0, 4, n_GPUs):
				lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
				with torch.no_grad():
					time_item_start = time.time()
					lr_batch = MacPI2SAI(lr_batch,angRes_in)
					sr_batch = net(lr_batch)
					time_total += time.time() - time_item_start 
				sr_batch = SAI2MacPI(sr_batch,angRes_out)
				sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))


			h, w = scale * h_size, scale * w_size
			h_half, w_half = h_size // 2, h_size // 2
			h_size, w_size = h_half , w_half 

			h_half, w_half = scale * h_half, scale * w_half
			h_size, w_size = scale * h_size, scale * w_size
		


			outLF = data.new(b, c, h, w)

			outLF[:, :, 0:h_half, 0:w_half] \
				= sr_list[0][:, :, 0:h_half, 0:w_half]
			outLF[:, :, 0:h_half, w_half:w] \
				= sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
			outLF[:, :, h_half:h, 0:w_half] \
				= sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
			outLF[:, :, h_half:h, w_half:w] \
				= sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

			# img_tmp = outLF.squeeze()
			# imgplot = plt.imshow(img_tmp.cpu().numpy())
			# plt.show()

			outLF = MacPI2SAI(outLF,angRes_out)

			# img_tmp = outLF.squeeze()
			# imgplot = plt.imshow(img_tmp.cpu().numpy())
			# plt.show()

			
		outLF = outLF.squeeze()

		#covert SAI to 4DLF
		outLF = SAI24DLF(outLF, cfg.angRes_out)
		label = SAI24DLF(label, cfg.angRes_out)
			
		psnr, ssim = cal_metrics_RE(label, outLF, cfg.angRes_in, cfg.angRes_out)

		psnr_iter_test.append(psnr)
		ssim_iter_test.append(ssim)

		if not (os.path.exists(cfg.save_path + '/' + test_name)):
			os.makedirs(cfg.save_path + '/' + test_name)
		scipy.io.savemat(cfg.save_path + '/' + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
						 {'LF': outLF.numpy()})
		pass


	psnr_epoch_test = float(np.array(psnr_iter_test).mean())
	ssim_epoch_test = float(np.array(ssim_iter_test).mean())

	return outLF, psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
	cfg = parse_args()
	test(cfg)