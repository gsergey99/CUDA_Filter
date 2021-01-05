all: 

	nvcc src/cuda_kernels.cu -o cuda_kernels -I/usr/include/opencv -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_datasets -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_line_descriptor -lopencv_optflow -lopencv_video -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_rgbd -lopencv_viz -lopencv_surface_matching -lopencv_text -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core

gpu-sobel:

	./cuda_kernels 0 "sobel" img/madrid.jpg	

gpu-sharpen:

	./cuda_kernels 0 "sharpen" img/gran_canaria.jpg

video-sobel:

	./cuda_kernels 1 "sobel" video/4k.mp4

video-sharpen:

	./cuda_kernels 1 "sharpen" video/4k.mp4

live-sobel:

	./cuda_kernels 2 "sobel" ""

live-sharpen:

	./cuda_kernels 2 "sharpen" ""

error:

	./cuda_kernels img/madrid.jpg ""
	
clean:

	rm cuda_kernels
