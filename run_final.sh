#this is the command for video processing with gfpgan only 
# python inference.py --checkpoint_path "/srv/MIRA_AI/checkpoints/Wav2Lip-SD-GAN.pth"  --face "/srv/MIRA_AI/input/Mira_Front.png" --audio "/srv/MIRA_AI/input/Mira.wav" --pads 0 25 0 0 --resize_factor 1 --face_det_batch_size 32 --wav2lip_batch_size 256

# cd scripts

# python face_restore_pipeline.py \
#   --input_video /srv/MIRA_AI/results/result_voice.mp4 \
#   --output_dir /srv/MIRA_AI/gfpgan_only \
#   --gfpgan_repo /srv/MIRA_AI/GFPGAN \
#   --only_center_face \
#-------------------------------------------------#----------------------------------------------------#

#blink effect
python demo_eye_motion.py \
  --input /srv/MIRA_AI/gfpgan_only/output_restored.mp4 \
  --output /srv/MIRA_AI/gfpgan_only/example_eye_motion_demo.mp4 \    # still in progress
  --duration 6 \
  --blink_at_sec 2 \
  --blink_len 7

#---------------------------------------------#----------------------------------------------------------#
#python video2frames.py --input_video "/srv/MIRA_AI/results/result_voice.mp4" --frames_path "/srv/MIRA_AI/frames_wav2lip"


#ffmpeg -r 25 -i "/srv/MIRA_AI/frames_hd/frame_%05d_out.jpg" -i "/srv/MIRA_AI/input/Mira.wav" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -crf 25 -preset veryslow -acodec copy "/srv/MIRA_AI/results/result_voice_hd.mkv"


# cd Real-ESRGAN

# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

# find /srv/MIRA_AI/frames_wav2lip -type f \( -iname '*.jpg' -o -iname '*.png' \) -print0 | \
# parallel -0 -j1 --will-cite '
#   python /srv/MIRA_AI/Real-ESRGAN/inference_realesrgan.py \
#     -n RealESRGAN_x2plus \
#     -i {} \
#     -o /srv/MIRA_AI/frames_hd \
#     -s 2.5 --face_enhance \
#     -t 1400 -g 0 --ext jpg --suffix out


#python inference_realesrgan.py -n RealESRGAN_x2plus -i "/srv/MIRA_AI/frames_wav2lip" --output "/srv/MIRA_AI/frames_hd" --outscale 2.5 --face_enhance --tile 0 -g 0 --ext jpg --suffix out this is my command

#ffmpeg -r 25 -start_number 0 -i "/srv/MIRA_AI/frames_hd/frame_%05d_out.jpg" -i "/srv/MIRA_AI/input/Mira.wav" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v h264_nvenc -preset p5 -cq 23 -pix_fmt yuv420p -c:a copy -shortest "/srv/MIRA_AI/results/result_voice_hd.mkv"

