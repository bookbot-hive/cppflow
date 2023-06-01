// Modified from: https://github.com/ZDisket/TensorVox

// CppFlow headers
#include <cppflow/cppflow.h>

// C++ headers
#include <iostream>

// AudioFile headers
#include "AudioFile.hpp"

typedef std::vector<std::tuple<std::string, cppflow::tensor>> TensorVec;

template <typename T>
struct TFTensor
{
    std::vector<T> Data;
    std::vector<int64_t> Shape;
    size_t TotalSize;
};

template <typename F>
TFTensor<F> CopyTensor(cppflow::tensor &InTens)
{
    std::vector<F> Data = InTens.get_data<F>();
    std::vector<int64_t> Shape = InTens.shape().get_data<int64_t>();
    size_t TotalSize = 1;
    for (const int64_t &Dim : Shape)
        TotalSize *= Dim;

    return TFTensor<F>{Data, Shape, TotalSize};
}

void ExportWAV(const std::string &Filename, const std::vector<float> &Data, unsigned SampleRate)
{
    AudioFile<float>::AudioBuffer Buffer;
    Buffer.resize(1);

    Buffer[0] = Data;
    size_t BufSz = Data.size();

    AudioFile<float> File;

    File.setAudioBuffer(Buffer);
    File.setAudioBufferSize(1, (int)BufSz);
    File.setNumSamplesPerChannel((int)BufSz);
    File.setNumChannels(1);
    File.setBitDepth(32);
    File.setSampleRate(SampleRate);

    File.save(Filename, AudioFileFormat::Wave);
}

int main()
{
    // folder where saved_model.pb is located
    cppflow::model lightspeech(std::string("fastspeech2"));
    cppflow::model mbmelgan(std::string("mbmelgan"));

    std::vector<int32_t> InputIDs = {8,
                                     31,
                                     12,
                                     43,
                                     7,
                                     34,
                                     53,
                                     13,
                                     18,
                                     50,
                                     14,
                                     17,
                                     30,
                                     7,
                                     12,
                                     15,
                                     18,
                                     42,
                                     18,
                                     42,
                                     50,
                                     17,
                                     51,
                                     14,
                                     17,
                                     42,
                                     16,
                                     12,
                                     53,
                                     17,
                                     16,
                                     12,
                                     53,
                                     17};

    // This is the shape of the input IDs, our equivalent to tf.expand_dims.
    std::vector<int64_t> InputIDShape = {1, (int64_t)InputIDs.size()};

    // Define the tensors
    cppflow::tensor input_ids{InputIDs, InputIDShape};
    cppflow::tensor energy_ratios{1.f};
    cppflow::tensor f0_ratios{1.f};
    // TODO: change speaker index here
    cppflow::tensor speaker_ids{0};
    cppflow::tensor speed_ratios{1.f};

    // Vector of input tensors
    TensorVec inputs = {{"serving_default_input_ids:0", input_ids},
                        {"serving_default_speaker_ids:0", speaker_ids},
                        {"serving_default_energy_ratios:0", energy_ratios},
                        {"serving_default_f0_ratios:0", f0_ratios},
                        {"serving_default_speed_ratios:0", speed_ratios}};

    // infer; LightSpeech returns 3 outputs: (mel, duration, pitch)
    auto outputs = lightspeech(inputs, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2"});
    // NOTE: FastSpeech2 returns >3 outputs!

    TFTensor<float> mel_spec = CopyTensor<float>(outputs[0]);
    TFTensor<int32_t> durations = CopyTensor<int32_t>(outputs[1]);

    // prepare mel spectrograms for input
    cppflow::tensor input_mels{mel_spec.Data, mel_spec.Shape};
    // infer
    auto out_audio = mbmelgan({{"serving_default_mels:0", input_mels}}, {"StatefulPartitionedCall:0"})[0];
    TFTensor<float> audio_tensor = CopyTensor<float>(out_audio);

    // write to file, specify sample rate
    ExportWAV("output.wav", audio_tensor.Data, 44100);

    return 0;
}