#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=train-clean-460
valid_set=dev-clean
test_sets="test-clean"

train_config=conf/train.yaml
# inference_config=conf/decode.yaml
inference_config=conf/tuning/decode_fastspeech_perturb_spk3.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio
vocoder_file=pretrain_model/libritts_hifigan.v1/checkpoint-400000steps.pkl

#    --download_model "	kan-bayashi/libritts_tts_train_xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss"
#    --download_model "kan-bayashi/libritts_tts_train_xvector_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave" \

./tts.sh \
    --ngpu 1 \
    --stage 7 \
    --stop_stage 7 \
    --lang en \
    --feats_type raw \
    --local_data_opts "${local_data_opts}" \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --use_xvector true \
    --token_type phn \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --download_model "kan-bayashi/libritts_tts_train_xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss" \
    --vocoder_file pretrained_model/libritts_hifigan.v1/checkpoint-2500000steps.pkl \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
