import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:porcupine_flutter/porcupine.dart'; 
import 'package:porcupine_flutter/porcupine_manager.dart';
import 'package:porcupine_flutter/porcupine_error.dart';
import 'package:flutter/services.dart';
// Add these imports for speech recognition
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:speech_to_text/speech_recognition_result.dart';
// Imports for tts
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/services.dart';

final AudioPlayer _audioPlayer = AudioPlayer();

class CameraScreen extends StatefulWidget {
  final CameraDescription camera;

  const CameraScreen({super.key, required this.camera});

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  final List<Map<String, dynamic>> _detectedObjectsWithDistance = [];
  Timer? _detectionTimer;
  final FlutterTts flutterTts = FlutterTts();
  PorcupineManager? _porcupineManager;
  bool _isListeningForWakeWord = false;
  static const String _accessKey = "WBnC2qh0vElEOqFKRwAnn8qSx6lqLaKKWJIGgPOvrn6OqXfomh+X/g==";
  
  // Add speech to text variables
  late stt.SpeechToText _speech;
  bool _isListening = false;
  String _userQuery = "";
  bool _isProcessingQuery = false;
  bool _hasSpokenIntroPrompt = false; // Track if "I'm listening" has been spoken
  bool _isProcessing = false;
  Timer? _processingFeedbackTimer;
  int _processingSeconds = 0;
  final AudioPlayer _audioPlayer = AudioPlayer(); // Create a single instance
  
  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializePorcupine();
    _initializeSpeechRecognition();
    _initializeAudio();
  }

  // Initialize audio player
  Future<void> _initializeAudio() async {
    try {
      // Load and cache processing sound
      await _audioPlayer.setReleaseMode(ReleaseMode.release);
    } catch (e) {
      print("Error initializing audio player: $e");
    }
  }

  void _playProcessingSound() async {
  try {
    // Light vibration for short feedback
    HapticFeedback.lightImpact();

    // Play "click" sound
    // await _audioPlayer.stop();
    // await _audioPlayer.play(AssetSource('sounds/click.mp3'));
    _playShortTTS("Got it working on it.");
  } catch (e) {
    print("Error in _playProcessingSound: $e");
  }
}

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.camera,  
      ResolutionPreset.medium,
    );
    _initializeControllerFuture = _controller.initialize().then((_) {
       // Set flash mode to off after camera is initialized
      _controller.setFlashMode(FlashMode.off);
      if (mounted) {
        setState(() {});
        _startObjectDetectionTimer();
      }
    });
  }

  // Initialize speech recognition
  Future<void> _initializeSpeechRecognition() async {
    _speech = stt.SpeechToText();
    await _speech.initialize(
      onStatus: (status) {
        print('Speech recognition status: $status');
        if (status == 'done' || status == 'notListening') {
          if (mounted) {
            setState(() {
              _isListening = false;
            });
          }
          if (_userQuery.isNotEmpty && !_isProcessingQuery) {
            _processUserQuery();
          }
        }
      },
      onError: (errorNotification) {
        print('Speech recognition error: $errorNotification');
        if (mounted) {
          setState(() {
            _isListening = false;
          });
          _showError('Error with speech recognition. Please try again.');
        }
        // Resume normal operation
        _startObjectDetectionTimer();
        _startWakeWordListener();
      },
    );
  }


  Future<void> _initializePorcupine() async {
    try {
      _porcupineManager = await PorcupineManager.fromBuiltInKeywords(
        _accessKey,
        [BuiltInKeyword.JARVIS],
        (keywordIndex) {
          print("Wake word detected at index: $keywordIndex");
          _handleWakeWordDetected();
        },
        errorCallback: (PorcupineException error) {
          print("Porcupine error: ${error.message}");
          // Try to restart wake word detection on error
          _restartWakeWordListener();
        },
      );

      // Start listening immediately
      _startWakeWordListener();
    } catch (e) {
      print("Error initializing Porcupine: $e");
      // Try to initialize again after a delay
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) {
          _initializePorcupine();
        }
      });
    }
  }

  void _startWakeWordListener() async {
    if (!_isListeningForWakeWord && _porcupineManager != null) {
      try {
        await _porcupineManager!.start();
        if (mounted) {
          setState(() {
            _isListeningForWakeWord = true;
          });
        }
        print("Listening for wake word...");
      } catch (e) {
        print("Error starting Porcupine: $e");
        _restartWakeWordListener();
      }
    }
  }

  void _stopWakeWordListener() async {
    if (_isListeningForWakeWord && _porcupineManager != null) {
      try {
        await _porcupineManager!.stop();
        if (mounted) {
          setState(() {
            _isListeningForWakeWord = false;
          });
        }
        print("Stopped listening for wake word.");
      } catch (e) {
        print("Error stopping Porcupine: $e");
      }
    }
  }

  // Function to restart wake word listener after errors
  void _restartWakeWordListener() {
    if (_porcupineManager != null) {
      try {
        _porcupineManager!.stop().then((_) {
          Future.delayed(const Duration(seconds: 1), () {
            if (mounted) {
              _startWakeWordListener();
            }
          });
        });
      } catch (e) {
        print("Error in restart wake word listener: $e");
        // Try to recreate the porcupine manager
        _porcupineManager?.delete();
        _porcupineManager = null;
        Future.delayed(const Duration(seconds: 2), () {
          if (mounted) {
            _initializePorcupine();
          }
        });
      }
    } else {
      if (mounted) {
        _initializePorcupine();
      }
    }
  }

  void _startProcessingFeedback() {
    // Reset counter
    _processingSeconds = 0;
    
    // Cancel any existing timer
    _processingFeedbackTimer?.cancel();
    
    if (mounted) {
      setState(() {
        _isProcessing = true;
      });
    }
    
    // Immediately provide initial feedback
    _playProcessingSound();
    
    // Set up timer for ongoing feedback
    _processingFeedbackTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      _processingSeconds += 1;
      
      // Always call the feedback functions directly
      // if (_processingSeconds <= 6) {
      //   _playProcessingSound();
      // } else {
      //   _playLongerWaitSound();
      // }
      
      // If processing is taking too long, provide verbal feedback
      if (_processingSeconds == 4) {
        _playShortTTS("Still working on it.");
      }
    });
  }

  void _stopProcessingFeedback() {
    _processingFeedbackTimer?.cancel();
    _processingFeedbackTimer = null;
    
    if (mounted) {
      setState(() {
        _isProcessing = false;
      });
    }
  }

 void _playLongerWaitSound() async {
  try {
    // Stronger haptic feedback
    HapticFeedback.mediumImpact();

    // Play "processing" sound
    // await _audioPlayer.stop();
    // await _audioPlayer.play(AssetSource('sounds/processing.mp3'));
    _playShortTTS("Hang on still thinking.");
  } catch (e) {
    print("Error in _playLongerWaitSound: $e");
  }
}

  // For brief TTS feedback that doesn't interrupt the main flow
  Future<void> _playShortTTS(String message) async {
    try {
      await flutterTts.setLanguage("hi-IN");
      await flutterTts.setSpeechRate(0.5);
      await flutterTts.setVolume(0.7);  // Slightly quieter than main TTS
      await flutterTts.speak(message);
    } catch (e) {
      print('Error with short TTS feedback: $e');
    }
  }

  @override
  void dispose() {
    _stopWakeWordListener();
    _stopObjectDetectionTimer();
    _porcupineManager?.delete();
    flutterTts.stop();
    _controller.dispose();
    _audioPlayer.dispose(); // Clean up audio player
    super.dispose();
    _speech.cancel(); // Make sure to cancel speech recognition
  }

  void _startObjectDetectionTimer() {
    _detectionTimer?.cancel(); // Cancel existing timer if any
    _detectionTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      _captureFrameAndDetect();
    });
  }

  void _stopObjectDetectionTimer() {
    _detectionTimer?.cancel();
    _detectionTimer = null;
  }

  // New method to handle wake word detection
  Future<void> _handleWakeWordDetected() async {
    // Make sure we don't process multiple activations
    if (_isListening || _isProcessingQuery) return;
    
    // Stop wake word detection and object detection temporarily
    _stopWakeWordListener();
    _stopObjectDetectionTimer();
    
    // Give haptic feedback to indicate activation
    HapticFeedback.mediumImpact();
    
    // Play a sound or give feedback that Jarvis is listening
    _hasSpokenIntroPrompt = true;
    await _speak("I'm listening");
    
    // Set a short delay to ensure TTS has finished completely
    await Future.delayed(const Duration(milliseconds: 800));
    
    // Start listening for user query
    await _startListening();
  }

  Future<void> _startListening() async {
    if (mounted) {
      setState(() {
        _userQuery = "";
        _isListening = true;
      });
    }
    
    try {
      await _speech.listen(
        onResult: _onSpeechResult,
        listenFor: const Duration(seconds: 30),
        pauseFor: const Duration(seconds: 8),
        partialResults: true,
        listenMode: stt.ListenMode.confirmation,
        onSoundLevelChange: (level) {
          // Could use sound level to provide feedback
        },
      );
    } catch (e) {
      print("Error starting speech recognition: $e");
      if (mounted) {
        _showError("Couldn't start listening. Please try again.");
      }
      // Resume normal operation
      _startObjectDetectionTimer();
      _startWakeWordListener();
    }
  }

  void _onSpeechResult(SpeechRecognitionResult result) {
    if (mounted) {
      setState(() {
        _userQuery = result.recognizedWords;
      });
    }
    
    print("Recognized speech: ${result.recognizedWords}, Final: ${result.finalResult}");
    
    if (result.finalResult && !_isProcessingQuery) {
      if (_userQuery.isNotEmpty) {
        // Give immediate acknowledgment that we heard the query
        _playShortTTS("Got it, thinking...");
        
        // Small delay before processing to let the acknowledgment play
        Future.delayed(const Duration(milliseconds: 500), () {
          _processUserQuery();
        });
      } else {
        _speak("I didn't catch that. Could you try again?");
        _startObjectDetectionTimer();
        _startWakeWordListener();
      }
    }
  }  

  // Update the processing function to avoid capturing "I'm listening"
  Future<void> _processUserQuery() async {
    if (_userQuery.isEmpty) {
      _speak("I didn't catch what you said. Please try again.");
      _startObjectDetectionTimer();
      _startWakeWordListener();
      return;
    }
    
    // Clean up any potential overlap with prompts
    String finalQuery = _userQuery;
    if (_hasSpokenIntroPrompt) {
      // Remove any accidental capture of "I'm listening" from the query
      final introWords = ["i'm listening", "i am listening"];
      for (var word in introWords) {
        if (finalQuery.toLowerCase().startsWith(word)) {
          finalQuery = finalQuery.substring(word.length).trim();
        }
      }
      _hasSpokenIntroPrompt = false;
    }
    
    if (mounted) {
      setState(() {
        _isProcessingQuery = true;
      });
    }
    
    // Start processing feedback
    _startProcessingFeedback();
    
    // Log the clean query we're about to send
    print("Processing clean query: $finalQuery");
    
    if (finalQuery.isNotEmpty) {
      await _captureFrameAndDetect(customQuery: finalQuery);
    } else {
      _speak("I didn't catch what you said. Please try again.");
    }
    
    // Stop processing feedback
    _stopProcessingFeedback();
    
    if (mounted) {
      setState(() {
        _isProcessingQuery = false;
        _userQuery = "";
      });
    }
    
    // Resume normal operation after processing
    Future.delayed(const Duration(seconds: 5), () {
      if (mounted) {
        _startObjectDetectionTimer();
        _startWakeWordListener();
      }
    });
  }

  Future<void> _captureFrameAndDetect({bool triggerSpeech = false, String? customQuery}) async {
    if (!_controller.value.isInitialized) {
      return;
    }
    
    if (!_controller.value.isTakingPicture) {
      try {
        final XFile image = await _controller.takePicture();
        final Uint8List bytes = await image.readAsBytes();
        final String base64Image = base64Encode(bytes);
        final detections = await _sendImageToBackend(base64Image);
        
        // If we have a custom query from speech or triggerSpeech is true
        if ((customQuery != null && customQuery.isNotEmpty) || (triggerSpeech && detections.isNotEmpty)) {
          // Pass the base64Image to the LLM function
          _askLLMForDescription(
            detections, 
            base64Image: base64Image, 
            query: customQuery ?? "Describe what you see."
          );
        }
      } catch (e) {
        print("Error capturing frame: $e");
      }
    }
  }

  Future<List<Map<String, dynamic>>> _sendImageToBackend(String base64Image) async {
    const backendUrl = 'http://192.168.193.42:8000/detect/';
    try {
      final response = await http.post(
        Uri.parse(backendUrl),
        headers: <String, String>{
          'Content-Type': 'application/json',
        },
        body: jsonEncode(<String, String>{
          'image': base64Image,
        }),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(response.body);
        final List<dynamic> detections = data['detections'];
        
        if (mounted) {
          setState(() {
            _detectedObjectsWithDistance.clear();
            for (var detection in detections) {
              _detectedObjectsWithDistance.add(detection as Map<String, dynamic>);
            }
          });
        }
        return _detectedObjectsWithDistance;
      } else {
        print('Failed to detect objects: ${response.statusCode}');
        print('Response body: ${response.body}');
        return [];
      }
    } catch (e) {
      print('Error sending image to backend: $e');
      return [];
    }
  }

  // Updated to accept a base64 image
  Future<void> _askLLMForDescription(
    List<Map<String, dynamic>> detections, 
    {String query = "Describe what you see.", 
    required String base64Image}) async {
    
    const backendUrl = 'http://192.168.193.42:8000/ask_llm/';

    if (detections.isNotEmpty) {
      try {
        final response = await http.post(
          Uri.parse(backendUrl),
          headers: <String, String>{
            'Content-Type': 'application/json',
          },
          body: jsonEncode(<String, dynamic>{
            'query': query,
            'detections': detections,
            'image': base64Image, // Pass the base64-encoded image
          }),
        );

        if (response.statusCode == 200) {
          final Map<String, dynamic> data = jsonDecode(response.body);
          final String answer = data['answer'];
          print('LLM Answer: $answer');
          _speak(answer);
        } else {
          print('Failed to get LLM response: ${response.statusCode}');
          print('Response body: ${response.body}');
          if (mounted) {
            _showError('Failed to get description.');
          }
        }
      } catch (e) {
        print('Error sending query to LLM: $e');
        if (mounted) {
          _showError('Error communicating with the server.');
        }
      }
    } else {
      _speak("I don't see anything to describe right now.");
    }
  }

  Future<void> _speak(String message) async {
    if (message.isEmpty) return;
    
    // Create a completer to track when speaking is done
    final Completer<void> speakingCompleter = Completer<void>();

    const ttsEndpoint = 'http://192.168.193.42:8000/text_to_speech/';

    try {
      if (mounted) {
        setState(() {
          // Show speaking indicator if needed
        });
      }

      final response = await http.post(
        Uri.parse(ttsEndpoint),
        headers: <String, String>{
          'Content-Type': 'application/json',
        },
        body: jsonEncode(<String, dynamic>{
          'text': message,
          'language': 'hi',
        }),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(response.body);
        final String audioBase64 = data['audio'];
        final String format = data['format'];

        final Uint8List audioBytes = base64Decode(audioBase64);

        final tempDir = await getTemporaryDirectory();
        final tempFile = File('${tempDir.path}/temp_audio.$format');
        await tempFile.writeAsBytes(audioBytes);

        final audioPlayer = AudioPlayer();
        // Add completion callback
        audioPlayer.onPlayerComplete.listen((_) {
          if (!speakingCompleter.isCompleted) {
            speakingCompleter.complete();
          }
        });
        await audioPlayer.play(DeviceFileSource(tempFile.path));
      } else {
        // Fallback to Flutter TTS
        await flutterTts.setLanguage("hi-IN");
        await flutterTts.setSpeechRate(0.4);
        await flutterTts.setVolume(1.0);
        await flutterTts.setPitch(1.0);
        
        // Set completion callback for Flutter TTS
        flutterTts.setCompletionHandler(() {
          if (!speakingCompleter.isCompleted) {
            speakingCompleter.complete();
          }
        });
        
        await flutterTts.speak(message);
      }
    } catch (e) {
      print('Error with TTS service: $e');
      if (mounted) {
        _showError('Error with speech service.');
      }
      
      // Ensure completer completes even on error
      if (!speakingCompleter.isCompleted) {
        speakingCompleter.complete();
      }
    }
    
    // Wait for speaking to complete before returning
    return speakingCompleter.future;
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.redAccent,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SeeU')),
      body: Column(
        children: <Widget>[
          Expanded(
            flex: 2,
            child: Stack(
              children: [
                FutureBuilder<void>(
                  future: _initializeControllerFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.done) {
                      return Container(
                        width: double.infinity,
                        height: double.infinity,
                        child: CameraPreview(_controller),
                      );
                    } else {
                      return const Center(child: CircularProgressIndicator());
                    }
                  },
                ),
                // Show listening indicator
                if (_isListening)
                  Positioned.fill(
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.3),
                      ),
                      child: Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(
                              Icons.mic,
                              color: Colors.white,
                              size: 48,
                            ),
                            const SizedBox(height: 16),
                            Text(
                              _userQuery.isEmpty ? "Listening..." : _userQuery,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 18,
                              ),
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                // Show processing indicator
                if (_isProcessing)
                  Positioned(
                    bottom: 20,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.7),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const SizedBox(
                              width: 24,
                              height: 24,
                              child: CircularProgressIndicator(
                                color: Colors.white,
                                strokeWidth: 2,
                              ),
                            ),
                            const SizedBox(width: 10),
                            Text(
                              "Processing${_getProcessingDots()}",
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          Expanded(
            flex: 1,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Detected Objects',
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                      // Status indicator
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: _isListeningForWakeWord ? Colors.green.withOpacity(0.2) : Colors.red.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              _isListeningForWakeWord ? Icons.mic : Icons.mic_off,
                              color: _isListeningForWakeWord ? Colors.green : Colors.red,
                              size: 16,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              _isListeningForWakeWord ? "Ready for 'Jarvis'" : "Wake word inactive",
                              style: TextStyle(
                                fontSize: 12,
                                color: _isListeningForWakeWord ? Colors.green : Colors.red,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Expanded(
                    child: ListView.builder(
                      itemCount: _detectedObjectsWithDistance.length,
                      itemBuilder: (context, index) {
                        final detection = _detectedObjectsWithDistance[index];
                        return Card(
                          margin: const EdgeInsets.symmetric(vertical: 4),
                          child: Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Text(
                                '${detection['object']} detected, ${detection['distance'] != null ? '${detection['distance']}m away' : 'distance unknown'}'),
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _handleWakeWordDetected, // For testing
                          icon: const Icon(Icons.mic),
                          label: const Text('Test Voice Input'),
                        ),
                      ),
                  //     const SizedBox(width: 8),
                  //     Expanded(
                  //       child: ElevatedButton.icon(
                  //         onPressed: () {
                  //           _playProcessingSound(); // Test haptic feedback
                  //         },
                  //         icon: const Icon(Icons.vibration),
                  //         label: const Text('Test Haptic'),
                  //       ),
                  //     ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (!_isListeningForWakeWord)
                    Center(
                      child: ElevatedButton.icon(
                        onPressed: _startWakeWordListener,
                        icon: const Icon(Icons.refresh),
                        label: const Text('Restart Wake Word Detection'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.amber,
                          foregroundColor: Colors.black,
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  // Helper method to generate animated dots for processing indicator
  String _getProcessingDots() {
    final dotsCount = (_processingSeconds ~/ 2) % 3 + 1;
    return "." * dotsCount;
  }
}

class NoCameraAvailableScreen extends StatelessWidget {
  const NoCameraAvailableScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Text('No cameras available on this device.'),
      ),
    );
  }
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.isNotEmpty ? cameras.first : null;

  runApp(
    MaterialApp(
      title: 'SeeU',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        useMaterial3: true,
      ),
      home: firstCamera != null
          ? CameraScreen(camera: firstCamera)
          : const NoCameraAvailableScreen(),
    ),
  );
}