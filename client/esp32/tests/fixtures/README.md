`webrtc-test-audio.raw` is the 8 kHz mono PCM16 test fixture from
[py-webrtcvad 2.0.10](https://github.com/wiseman/py-webrtcvad/blob/2.0.10/test-audio.raw),
commit `ba13ceea9dd3f3a068e869b2e94413205bd87f7d`, distributed under the
[upstream license](../../components/webrtc_vad/LICENSE). It is public upstream test
data, not a household recording. The host test repeats each sample to feed the
16 kHz frontend; this tests detector integration, not acoustic/ASR quality.
