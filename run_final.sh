# python inference.py --checkpoint_path "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\checkpoints\Wav2Lip-SD-GAN.pt"  --face "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\input\Mira_Front.png" --audio "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\input\Mira.wav" --pads 0 20 0 0 --resize_factor 720 --face_det_batch_size 32 --wav2lip_batch_size 256


# python video2frames.py --input_video "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\results\result_voice.mp4" --frames_path "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\frames_wav2lip"
# cd Real-ESRGAN
# python inference_realesrgan.py -n RealESRGAN_x2plus -i "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\frames_wav2lip" --output "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\frames_hd" --outscale 2.5 --face_enhance --tile 400 --tile_pad 10
#ffmpeg -r 25 -i "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\frames_hd\frame_%05d_out.jpg" -i "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\input\Mira.wav" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -crf 25 -preset veryslow -acodec copy "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\results\result_voice_hd.mkv"

ffmpeg -r 25 -start_number 0 -i "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\frames_hd\frame_%05d_out.jpg" -i "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\input\Mira.wav" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v h264_nvenc -preset p5 -cq 23 -pix_fmt yuv420p -c:a copy -shortest "C:\Users\AAAA\Desktop\og wav2lip\Wav2Lip\results\result_voice_hd.mkv"

