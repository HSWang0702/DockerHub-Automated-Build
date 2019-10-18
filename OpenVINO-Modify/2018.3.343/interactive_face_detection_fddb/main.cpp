/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine interactive_face_detection sample application
* \file interactive_face_detection_sample/main.cpp
* \example interactive_face_detection_sample/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <regex>
#include <typeinfo>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "interactive_face_detection.hpp"
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <ext_list.hpp>

#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_n_ag < 1) {
        throw std::logic_error("Parameter -n_ag cannot be 0");
    }

    if (FLAGS_n_hp < 1) {
        throw std::logic_error("Parameter -n_hp cannot be 0");
    }

    // no need to wait for a key press from a user if an output image/video file is not shown.
    FLAGS_no_wait |= FLAGS_no_show;

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
    ExecutableNetwork net;
    InferencePlugin * plugin;
    InferRequest::Ptr request;
    std::string & commandLineFlag;
    std::string topoName;
    const int maxBatch;
    const bool isAsync;

    BaseDetection(std::string &commandLineFlag, std::string topoName, int maxBatch, bool isAsync = false)
        : commandLineFlag(commandLineFlag), topoName(topoName), maxBatch(maxBatch), isAsync(isAsync) {
            if (isAsync) {
                slog::info << "Use async mode for " << topoName << slog::endl;
            }
        }

    virtual ~BaseDetection() {}

    ExecutableNetwork* operator ->() {
        return &net;
    }
    virtual CNNNetwork read()  = 0;

    virtual void submitRequest() {
        if (!enabled() || request == nullptr) return;
        if (isAsync) {
            request->StartAsync();
        } else {
            request->Infer();
        }
    }

    virtual void wait() {
        if (!enabled()|| !request || !isAsync) return;
        request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
    void printPerformanceCounts() {
        if (!enabled()) {
            return;
        }
        slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
        ::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
    }
};

struct FaceDetectionClass : BaseDetection {
    std::string input;
    std::string output;
    int maxProposalCount;
    int objectSize;
    int enquedFrames = 0;
    float width = 0;
    float height = 0;
    bool resultsFetched = false;
    std::vector<std::string> labels;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        if (!enquedFrames) return;
        enquedFrames = 0;
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void enqueue(const cv::Mat &frame) {
        if (!enabled()) return;

        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        width = frame.cols;
        height = frame.rows;

        Blob::Ptr  inputBlob = request->GetBlob(input);

        matU8ToBlob<uint8_t>(frame, inputBlob);

        enquedFrames = 1;
    }


    FaceDetectionClass() : BaseDetection(FLAGS_m, "Face Detection", 1, FLAGS_async) {}
    CNNNetwork read() override {
        slog::info << "Loading network files for Face Detection" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is set to " << maxBatch << slog::endl;
        netReader.getNetwork().setBatchSize(maxBatch);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Read labels (if any)**/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";

        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Face Detection inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one input");
        }
        InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Face Detection outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one output");
        }
        DataPtr& _output = outputInfo.begin()->second;
        output = outputInfo.begin()->first;

        const CNNLayerPtr outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
        if (outputLayer->type != "DetectionOutput") {
            throw std::logic_error("Face Detection network output layer(" + outputLayer->name +
                ") should be DetectionOutput, but was " +  outputLayer->type);
        }

        if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
            throw std::logic_error("Face Detection network output layer (" +
                output + ") should have num_classes integer attribute");
        }

        const int num_classes = outputLayer->GetParamAsInt("num_classes");
        if (labels.size() != num_classes) {
            if (labels.size() == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else
                labels.clear();
        }
        const SizeVector outputDims = _output->getTensorDesc().getDims();
        maxProposalCount = outputDims[2];
        objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
                                           std::to_string(outputDims.size()));
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        slog::info << "Loading Face Detection model to the "<< FLAGS_d << " plugin" << slog::endl;
        input = inputInfo.begin()->first;
        return netReader.getNetwork();
    }

    void fetchResults() {
        if (!enabled()) return;
        results.clear();
        if (resultsFetched) return;
        resultsFetched = true;
        const float *detections = request->GetBlob(output)->buffer().as<float *>();

        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];
            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];
            if (r.confidence <= FLAGS_t) {
                continue;
            }

            r.location.x = detections[i * objectSize + 3] * width;
            r.location.y = detections[i * objectSize + 4] * height;
            r.location.width = detections[i * objectSize + 5] * width - r.location.x;
            r.location.height = detections[i * objectSize + 6] * height - r.location.y;

            if (image_id < 0) {
                break;
            }
            if (FLAGS_r) {
                std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                          << r.location.height << ")"
                          << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
            }

            results.push_back(r);
        }
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferencePlugin & plg, bool enable_dynamic_batch = false) const {
        if (detector.enabled()) {
            std::map<std::string, std::string> config;
            if (enable_dynamic_batch) {
                config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
            }
            detector.net = plg.LoadNetwork(detector.read(), config);
            detector.plugin = &plg;
        }
    }
};


struct CallStat {
    public:
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    double getSmoothedDuration() {
        // Additional check is needed for the first frame while duration of the first
        // visualisation is not calculated yet.
        if (_smoothed_duration < 0) {
            auto t = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<ms>(t - _last_call_start).count();
        }
        return _smoothed_duration;
    }

    double getTotalDuration() {
        return _total_duration;
    }

    void calculateDuration() {
        auto t = std::chrono::high_resolution_clock::now();
        _last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
        _number_of_calls++;
        _total_duration += _last_call_duration;
        if (_smoothed_duration < 0) {
            _smoothed_duration = _last_call_duration;
        }
        double alpha = 0.1;
        _smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
    }

    void setStartTime() {
        _last_call_start = std::chrono::high_resolution_clock::now();
    }

    private:
    size_t _number_of_calls {0};
    double _total_duration {0.0};
    double _last_call_duration {0.0};
    double _smoothed_duration {-1.0};
    std::chrono::time_point<std::chrono::high_resolution_clock> _last_call_start;
};

class Timer {
    public:
    void start(const std::string& name) {
        if (_timers.find(name) == _timers.end()) {
            _timers[name] = CallStat();
        }
        _timers[name].setStartTime();
    }

    void finish(const std::string& name) {
        auto& timer = (*this)[name];
        timer.calculateDuration();
    }

    CallStat& operator[](const std::string& name) {
        if (_timers.find(name) == _timers.end()) {
            throw std::logic_error("No timer with name " + name + ".");
        }
        return _timers[name];
    }

    private:
    std::map<std::string, CallStat> _timers;
};

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::regex Imagesets_string("ImageSets/Main/.*");
        std::string JPEGImages_string("JPEGImages/");
        std::string root_dir = std::regex_replace(FLAGS_i.c_str(), Imagesets_string, JPEGImages_string);

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;

        std::ifstream ground_truth;
        ground_truth.open(FLAGS_i);
        std::string image_name;
	    std::string image;
        std::vector<double> detection_times;

        //Write results
        std::string output_file;
        std::ofstream detection_output;
        if (FLAGS_no_show) {
            output_file = FLAGS_output_dir.c_str();
            output_file.append(FLAGS_output_name).append(".txt");
            detection_output.open(output_file);
        }

        if (ground_truth.is_open()) {
            getline (ground_truth, image_name);
            if (image_name == "") {
                throw std::logic_error("Failed to get image");
            }
	        // image = root_dir;
	        image = image_name;
            cap.open(image);

            const size_t width  = (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH);
            const size_t height = (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT);

            // read input (video) frame
            cv::Mat frame;
            if (!cap.read(frame)) {
                throw std::logic_error("Failed to get frame from cv::VideoCapture");
            }
            // -----------------------------------------------------------------------------------------------------
            // --------------------------- 1. Load Plugin for inference engine -------------------------------------
            std::map<std::string, InferencePlugin> pluginsForDevices;
            std::vector<std::pair<std::string, std::string>> cmdOptions = {
                {FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp},
                {FLAGS_d_em, FLAGS_m_em}
            };

            FaceDetectionClass FaceDetection;

            for (auto && option : cmdOptions) {
                auto deviceName = option.first;
                auto networkName = option.second;

                if (deviceName == "" || networkName == "") {
                    continue;
                }

                if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                    continue;
                }
                slog::info << "Loading plugin " << deviceName << slog::endl;
                InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

                /** Printing plugin version **/
                printPluginVersion(plugin, std::cout);

                /** Load extensions for the CPU plugin **/
                if ((deviceName.find("CPU") != std::string::npos)) {
                    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                    if (!FLAGS_l.empty()) {
                        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                        auto extension_ptr = make_so_pointer<MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);
                        plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
                    }
                } else if (!FLAGS_c.empty()) {
                    // Load Extensions for other plugins not CPU
                    plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
                }
                pluginsForDevices[deviceName] = plugin;
            }

            /** Per layer metrics **/
            if (FLAGS_pc) {
                for (auto && plugin : pluginsForDevices) {
                    plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
                }
            }
            // -----------------------------------------------------------------------------------------------------

            // --------------------------- 2. Read IR models and load them to plugins ------------------------------
            // Disable dynamic batching for face detector as long it processes one image at a time.
            Load(FaceDetection).into(pluginsForDevices[FLAGS_d], false);
            // -----------------------------------------------------------------------------------------------------

            // --------------------------- 3. Do inference ---------------------------------------------------------
            // Start inference & calc performance.
            slog::info << "Start inference " << slog::endl;
            if (!FLAGS_no_show) {
                std::cout << "Press any key to stop" << std::endl;
            }

            Timer timer;
            timer.start("total");

            std::ostringstream out;
            size_t framesCounter = 0;
            bool frameReadStatus;
            bool isLastFrame;
            cv::Mat prev_frame, next_frame;

            // Detect all faces on the first frame and read the next one.
            timer.start("detection");
            FaceDetection.enqueue(frame);
            FaceDetection.submitRequest();
            timer.finish("detection");

            detection_output << image << std::endl;

            prev_frame = frame.clone();

            image_name.clear();
            getline (ground_truth, image_name);
            // image = root_dir;
            image = image_name;

            // Read next frame.
            framesCounter++;
            timer.start("video frame decoding");
            cap.open(image);
            frameReadStatus = cap.read(frame);
            timer.finish("video frame decoding");

            while (true) {
                framesCounter++;
                isLastFrame = !frameReadStatus;

                timer.start("detection");
                // Retrieve face detection results for previous frame.
                FaceDetection.wait();
                FaceDetection.fetchResults();
                auto prev_detection_results = FaceDetection.results;

                // No valid frame to infer if previous frame is last.
                if (!isLastFrame) {
                    FaceDetection.enqueue(frame);
                    FaceDetection.submitRequest();
                }
                timer.finish("detection");

                if (FLAGS_no_show) {
                    // std::cout << "Face detection time: " << std::fixed << std::setprecision(2) << timer["detection"].getSmoothedDuration() << " ms" << std::endl;
                    detection_output << prev_detection_results.size() << std::endl;
                    for (auto &result : prev_detection_results) {
                        cv::Rect rect = result.location;
                        detection_output << int(result.location.x) << " " << int(result.location.y) << " " << int(result.location.x + result.location.width) << " " << int(result.location.y + result.location.height) << " " << std::fixed << std::setprecision(3) << floor(result.confidence*1000)/1000 << std::endl;
                    }

                    if (frameReadStatus) {
                        detection_output << image << std::endl; // Next frame name
                    }
                }

                // Read next frame if current one is not last.
                if (!isLastFrame) {
                    image_name.clear();
                    getline (ground_truth, image_name);
                    if (image_name == "") {
                        frameReadStatus = false;
                    } else {
                        // image = root_dir;
                        image = image_name;

                        timer.start("video frame decoding");
                        cap.open(image);
                        frameReadStatus = cap.read(next_frame);
                        timer.finish("video frame decoding");
                    }
                }

                // Visualize results.
                if (!FLAGS_no_show) {
                    timer.start("visualization");
                    out.str("");
                    out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                        << (timer["video frame decoding"].getSmoothedDuration() +
                            timer["visualization"].getSmoothedDuration())
                        << " ms";
                    cv::putText(prev_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                                cv::Scalar(255, 0, 0));

                    out.str("");
                    out << "Face detection time: " << std::fixed << std::setprecision(2)
                        << timer["detection"].getSmoothedDuration()
                        << " ms ("
                        << 1000.f /
                        (timer["detection"].getSmoothedDuration())
                        << " fps)";
                    cv::putText(prev_frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                                cv::Scalar(255, 0, 0));

                    // For every detected face.
                    for (auto &result : prev_detection_results) {
                        cv::Rect rect = result.location;

                        out.str("");
                        out << "face" << ": " << std::fixed << std::setprecision(3) << result.confidence;

                        cv::putText(prev_frame,
                                    out.str(),
                                    cv::Point2f(result.location.x, result.location.y - 15),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.8,
                                    cv::Scalar(0, 0, 255));

                        auto genderColor = cv::Scalar(80, 80, 80);
                        cv::rectangle(prev_frame, result.location, genderColor, 2);
                    }

                    cv::imshow("Detection results", prev_frame);
                    timer.finish("visualization");
                    cv::waitKey(0);
                }

                detection_times.push_back(timer["detection"].getSmoothedDuration());

                // End of file (or a single frame file like an image). We just keep last frame displayed to let user check what was shown
                if (isLastFrame) {
                    timer.finish("total");
                    if (!FLAGS_no_wait) {
                        std::cout << "No more frames to process. Press any key to exit" << std::endl;
                        cv::waitKey(0);
                    }
                    break;
                } else if (!FLAGS_no_show && -1 != cv::waitKey(1)) {
                    timer.finish("total");
                    break;
                }

                prev_frame = frame;
                frame = next_frame;
                next_frame = cv::Mat();
            }
            
            double detections_sum = std::accumulate(detection_times.begin(), detection_times.end(), 0.0);
            double detections_mean = detections_sum / detection_times.size();

            double diff_accumulate = 0.0;
            std::for_each (std::begin(detection_times), std::end(detection_times), [&](const double each_detection) {
                diff_accumulate += (each_detection - detections_mean) * (each_detection - detections_mean);
            });
            double detections_stdev = sqrt(diff_accumulate / (detection_times.size()-1));

            framesCounter = framesCounter - 1; // last time don't count
            slog::info << "Number of processed frames: " << framesCounter << slog::endl;
            slog::info << "Mean of detection times : " << detections_mean << slog::endl;
            slog::info << "Stdev of detection times: " << detections_stdev << slog::endl;
            slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

            if (FLAGS_no_show) {
                detection_output.close();
            }

            // Show performace results.
            if (FLAGS_pc) {
                FaceDetection.printPerformanceCounts();
            }
            // -----------------------------------------------------------------------------------------------------
        ground_truth.close();
        }
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}

