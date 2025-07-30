"""
Speech-to-Text (STT) Server with Real-Time Transcription and WebSocket Interface

This server provides real-time speech-to-text (STT) transcription using the RealtimeSTT library. It allows clients to connect via WebSocket to send audio data and receive real-time transcription updates. The server supports configurable audio recording parameters, voice activity detection (VAD), and wake word detection. It is designed to handle continuous transcription as well as post-recording processing, enabling real-time feedback with the option to improve final transcription quality after the complete sentence is recognized.

### Features:
- Real-time transcription using pre-configured or user-defined STT models.
- WebSocket-based communication for control and data handling.
- Flexible recording and transcription options, including configurable pauses for sentence detection.
- Supports Silero and WebRTC VAD for robust voice activity detection.
- Production-ready with support for thousands of concurrent calls.
- Self-healing architecture with no hard-coded delays.

### Starting the Server:
You can start the server using the command-line interface (CLI) command `stt-server`, passing the desired configuration options.

```bash
stt-server [OPTIONS]
```

### Available Parameters:
    - `-m, --model`: Model path or size; default 'large-v2'.
    - `-r, --rt-model, --realtime_model_type`: Real-time model size; default 'tiny.en'.
    - `-l, --lang, --language`: Language code for transcription; default 'en'.
    - `-i, --input-device, --input_device_index`: Audio input device index; default 1.
    - `-c, --control, --control_port`: WebSocket control port; default 8011.
    - `-d, --data, --data_port`: WebSocket data port; default 8012.
    - `-w, --wake_words`: Wake word(s) to trigger listening; default "".
    - `-D, --debug`: Enable debug logging.
    - `-W, --write`: Save audio to WAV file.
    - `-s, --silence_timing`: Enable dynamic silence duration for sentence detection; default True. 
    - `-b, --batch, --batch_size`: Batch size for inference; default 16.
    - `--root, --download_root`: Specifies the root path were the Whisper models are downloaded to.
    - `--silero_sensitivity`: Silero VAD sensitivity (0-1); default 0.05.
    - `--silero_use_onnx`: Use Silero ONNX model; default False.
    - `--webrtc_sensitivity`: WebRTC VAD sensitivity (0-3); default 3.
    - `--min_length_of_recording`: Minimum recording duration in seconds; default 1.1.
    - `--min_gap_between_recordings`: Min time between recordings in seconds; default 0.
    - `--enable_realtime_transcription`: Enable real-time transcription; default True.
    - `--realtime_processing_pause`: Pause between audio chunk processing; default 0.02.
    - `--silero_deactivity_detection`: Use Silero for end-of-speech detection; default True.
    - `--early_transcription_on_silence`: Start transcription after silence in seconds; default 0.2.
    - `--beam_size`: Beam size for main model; default 5.
    - `--beam_size_realtime`: Beam size for real-time model; default 3.
    - `--init_realtime_after_seconds`: Initial waiting time for realtime transcription; default 0.2.
    - `--realtime_batch_size`: Batch size for the real-time transcription model; default 16.
    - `--initial_prompt`: Initial main transcription guidance prompt.
    - `--initial_prompt_realtime`: Initial realtime transcription guidance prompt.
    - `--end_of_sentence_detection_pause`: Silence duration for sentence end detection; default 0.45.
    - `--unknown_sentence_detection_pause`: Pause duration for incomplete sentence detection; default 0.7.
    - `--mid_sentence_detection_pause`: Pause for mid-sentence break; default 2.0.
    - `--wake_words_sensitivity`: Wake word detection sensitivity (0-1); default 0.5.
    - `--wake_word_timeout`: Wake word timeout in seconds; default 5.0.
    - `--wake_word_activation_delay`: Delay before wake word activation; default 20.
    - `--wakeword_backend`: Backend for wake word detection; default 'none'.
    - `--openwakeword_model_paths`: Paths to OpenWakeWord models.
    - `--openwakeword_inference_framework`: OpenWakeWord inference framework; default 'tensorflow'.
    - `--wake_word_buffer_duration`: Wake word buffer duration in seconds; default 1.0.
    - `--use_main_model_for_realtime`: Use main model for real-time transcription.
    - `--use_extended_logging`: Enable extensive log messages.
    - `--logchunks`: Log incoming audio chunks.
    - `--compute_type`: Type of computation to use.
    - `--input_device_index`: Index of the audio input device.
    - `--gpu_device_index`: Index of the GPU device.
    - `--device`: Device to use for computation.
    - `--handle_buffer_overflow`: Handle buffer overflow during transcription.
    - `--suppress_tokens`: Suppress tokens during transcription.
    - `--allowed_latency_limit`: Allowed latency limit for real-time transcription.
    - `--faster_whisper_vad_filter`: Enable VAD filter for Faster Whisper; default False.


### WebSocket Interface:
The server supports two WebSocket connections:
1. **Control WebSocket**: Used to send and receive commands, such as setting parameters or calling recorder methods.
2. **Data WebSocket**: Used to send audio data for transcription and receive real-time transcription updates.

The server will broadcast real-time transcription updates to all connected clients on the data WebSocket.
"""

from .install_packages import check_and_install_packages
from difflib import SequenceMatcher
from collections import deque
from datetime import datetime
import logging
import asyncio
import pyaudio
import base64
import sys
import time
import weakref
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading

# Production-ready metrics and monitoring
@dataclass
class ServerMetrics:
    """Production metrics for monitoring server health and performance"""
    active_connections: int = 0
    total_connections: int = 0
    audio_chunks_processed: int = 0
    transcription_errors: int = 0
    recorder_errors: int = 0
    last_activity: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    
    def reset_activity(self):
        self.last_activity = time.time()
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time
    
    def get_connections_per_minute(self) -> float:
        uptime = self.get_uptime()
        return (self.total_connections / uptime) * 60 if uptime > 0 else 0

class RecorderState(Enum):
    """Recorder state enumeration for better state management"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"

@dataclass
class RecorderHealth:
    """Health monitoring for the recorder"""
    state: RecorderState = RecorderState.INITIALIZING
    last_successful_processing: float = field(default_factory=time.time)
    consecutive_errors: int = 0
    error_threshold: int = 5
    recovery_attempts: int = 0
    max_recovery_attempts: int = 10
    
    def record_success(self):
        self.state = RecorderState.READY
        self.last_successful_processing = time.time()
        self.consecutive_errors = 0
        self.recovery_attempts = 0
    
    def record_error(self):
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.error_threshold:
            self.state = RecorderState.ERROR
    
    def start_recovery(self):
        self.state = RecorderState.RECOVERING
        self.recovery_attempts += 1
    
    def can_recover(self) -> bool:
        return self.recovery_attempts < self.max_recovery_attempts

# Global state management
debug_logging = False
extended_logging = False
send_recorded_chunk = False
log_incoming_chunks = False
silence_timing = False
writechunks = False
wav_file = None

# Production metrics
server_metrics = ServerMetrics()
recorder_health = RecorderHealth()

hard_break_even_on_background_noise = 3.0
hard_break_even_on_background_noise_min_texts = 3
hard_break_even_on_background_noise_min_similarity = 0.99
hard_break_even_on_background_noise_min_chars = 15

text_time_deque = deque()
loglevel = logging.WARNING

FORMAT = pyaudio.paInt16
CHANNELS = 1

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

check_and_install_packages([
    {
        'module_name': 'RealtimeSTT',                 # Import module
        'attribute': 'AudioToTextRecorder',           # Specific class to check
        'install_name': 'RealtimeSTT',                # Package name for pip install
    },
    {
        'module_name': 'websockets',                  # Import module
        'install_name': 'websockets',                 # Package name for pip install
    },
    {
        'module_name': 'numpy',                       # Import module
        'install_name': 'numpy',                      # Package name for pip install
    },
    {
        'module_name': 'scipy.signal',                # Submodule of scipy
        'attribute': 'resample',                      # Specific function to check
        'install_name': 'scipy',                      # Package name for pip install
    }
])

# Define ANSI color codes for terminal output
class bcolors:
    HEADER = '\033[95m'   # Magenta
    OKBLUE = '\033[94m'   # Blue
    OKCYAN = '\033[96m'   # Cyan
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'     # Red
    ENDC = '\033[0m'      # Reset to default
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f"{bcolors.BOLD}{bcolors.OKCYAN}Starting production-ready STT server, please wait...{bcolors.ENDC}")

# Initialize colorama
from colorama import init, Fore, Style
init()

from RealtimeSTT import AudioToTextRecorder
from scipy.signal import resample
import numpy as np
import websockets
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
import aiohttp
from aiohttp import web
import logging
import wave
import json

# Production-ready hybrid recorder manager (thread-based recorder with async communication)
class HybridRecorderManager:
    """Hybrid recorder manager with thread-based recorder and async communication"""
    
    def __init__(self, config: dict, loop: asyncio.AbstractEventLoop):
        self.config = config
        self.loop = loop
        self.recorder: Optional[AudioToTextRecorder] = None
        self.recorder_thread: Optional[threading.Thread] = None
        self.health_task: Optional[asyncio.Task] = None
        self.ready_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        self.processing_lock = asyncio.Lock()
        self.recorder_ready = threading.Event()
        self.stop_recorder = False
        
        # Thread-safe queues for communication
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._last_activity_check = time.time()
        
    async def initialize(self):
        """Initialize the recorder in a separate thread"""
        try:
            recorder_health.state = RecorderState.INITIALIZING
            print(f"{bcolors.OKGREEN}Initializing production-ready RealtimeSTT server with parameters:{bcolors.ENDC}")
            for key, value in self.config.items():
                print(f"    {bcolors.OKBLUE}{key}{bcolors.ENDC}: {value}")
            
            # Start recorder in separate thread
            self.recorder_thread = threading.Thread(target=self._recorder_thread_worker, daemon=True)
            self.recorder_thread.start()
            
            # Wait for recorder to be ready
            await asyncio.to_thread(self.recorder_ready.wait)
            
            print(f"{bcolors.OKGREEN}{bcolors.BOLD}RealtimeSTT initialized successfully{bcolors.ENDC}")
            recorder_health.record_success()
            self.ready_event.set()
            
            # Start monitoring tasks
            self.health_task = asyncio.create_task(self._health_monitor())
            
        except Exception as e:
            print(f"{bcolors.FAIL}[ERROR] Failed to initialize recorder: {e}{bcolors.ENDC}")
            recorder_health.record_error()
            self.error_event.set()
            raise
    
    def _recorder_thread_worker(self):
        """Worker thread for the recorder - runs the blocking recorder operations"""
        try:
            # Create recorder in this thread
            self.recorder = AudioToTextRecorder(**self.config)
            
            # Add reset method to recorder object
            def reset_recorder_state_method():
                self._reset_recorder_state()
            
            self.recorder.reset_recorder_state = reset_recorder_state_method
            
            # Signal that recorder is ready
            self.recorder_ready.set()
            
            # Main processing loop
            def process_text(full_sentence):
                # Put result in queue for async processing
                self.result_queue.put(('full_sentence', full_sentence))
            
            while not self.stop_recorder:
                try:
                    # Process text - this is the blocking operation
                    self.recorder.text(process_text)
                    recorder_health.record_success()
                    server_metrics.reset_activity()
                    
                    # Update activity time to prevent stuck detection
                    if hasattr(self, '_last_activity_check'):
                        self._last_activity_check = time.time()
                    
                except Exception as e:
                    print(f"{bcolors.FAIL}[ERROR] Recorder thread error: {e}{bcolors.ENDC}")
                    server_metrics.recorder_errors += 1
                    recorder_health.record_error()
                    # Don't sleep, just continue processing
                    
        except Exception as e:
            print(f"{bcolors.FAIL}[ERROR] Failed to create recorder: {e}{bcolors.ENDC}")
            recorder_health.record_error()
    
    async def _health_monitor(self):
        """Monitor recorder health without blocking"""
        last_activity_check = time.time()
        stuck_threshold = 10  # 10 seconds without activity
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Check if recorder is responsive
                if recorder_health.state == RecorderState.ERROR:
                    await self._attempt_recovery()
                
                # Check if recorder is stuck (no activity for too long)
                current_time = time.time()
                if hasattr(self, '_last_activity_check'):
                    if current_time - self._last_activity_check > stuck_threshold:
                        if recorder_health.state == RecorderState.READY:
                            print(f"{bcolors.WARNING}[DEBUG] Recorder appears stuck, attempting recovery{bcolors.ENDC}")
                            recorder_health.record_error()
                            await self._attempt_recovery()
                        self._last_activity_check = current_time
                    
                # Process any results from the recorder thread
                await self._process_recorder_results()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"{bcolors.WARNING}[DEBUG] Health monitor error: {e}{bcolors.ENDC}")
    
    async def _process_recorder_results(self):
        """Process results from the recorder thread"""
        try:
            while not self.result_queue.empty():
                result_type, data = self.result_queue.get_nowait()
                
                if result_type == 'full_sentence':
                    await self._handle_full_sentence(data)
                    
        except queue.Empty:
            pass
        except Exception as e:
            print(f"{bcolors.FAIL}[ERROR] Error processing recorder results: {e}{bcolors.ENDC}")
    
    async def _attempt_recovery(self):
        """Attempt to recover from errors without blocking"""
        if not recorder_health.can_recover():
            print(f"{bcolors.FAIL}[ERROR] Max recovery attempts reached{bcolors.ENDC}")
            return
        
        try:
            recorder_health.start_recovery()
            print(f"{bcolors.WARNING}[DEBUG] Attempting recorder recovery (attempt {recorder_health.recovery_attempts}){bcolors.ENDC}")
            
            # Try to clear audio queue and reset state
            if self.recorder:
                await asyncio.to_thread(self.recorder.clear_audio_queue)
                self._reset_recorder_state()
            
            recorder_health.record_success()
            print(f"{bcolors.OKGREEN}[DEBUG] Recorder recovery successful{bcolors.ENDC}")
            
        except Exception as e:
            print(f"{bcolors.FAIL}[ERROR] Recovery attempt failed: {e}{bcolors.ENDC}")
            recorder_health.record_error()
    
    def _reset_recorder_state(self):
        """Reset recorder state without blocking"""
        try:
            if self.recorder:
                self.recorder.post_speech_silence_duration = global_args.unknown_sentence_detection_pause
        except Exception as e:
            print(f"{bcolors.WARNING}[DEBUG] Error in reset_recorder_state: {e}{bcolors.ENDC}")
    
    async def process_audio(self, audio_data: bytes):
        """Process audio data asynchronously"""
        if not self.recorder or recorder_health.state == RecorderState.ERROR:
            return
        
        try:
            # Put audio in queue for recorder thread
            self.audio_queue.put(audio_data)
            
            # Feed audio to recorder
            await asyncio.to_thread(self.recorder.feed_audio, audio_data)
            server_metrics.audio_chunks_processed += 1
            server_metrics.reset_activity()
                
        except Exception as e:
            print(f"{bcolors.FAIL}[ERROR] Error processing audio: {e}{bcolors.ENDC}")
            server_metrics.transcription_errors += 1
            recorder_health.record_error()
    
    async def _handle_full_sentence(self, full_sentence: str):
        """Handle full sentence processing asynchronously"""
        try:
            full_sentence = preprocess_text(full_sentence)
            message = json.dumps({
                'type': 'fullSentence',
                'text': full_sentence
            })
            
            await audio_queue.put(message)
            
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            if extended_logging:
                print(f"  [{timestamp}] Full text: {bcolors.BOLD}Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}\n", flush=True, end="")
            else:
                print(f"\r[{timestamp}] {bcolors.BOLD}Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}\n")
            
            # Reset only the text variable, not the entire recorder state
            global prev_text
            prev_text = ""
            
        except Exception as e:
            print(f"{bcolors.FAIL}[ERROR] Error handling full sentence: {e}{bcolors.ENDC}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.shutdown_event.set()
        self.stop_recorder = True
        
        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except asyncio.CancelledError:
                pass
        
        if self.recorder:
            try:
                await asyncio.to_thread(self.recorder.shutdown)
            except Exception as e:
                print(f"{bcolors.WARNING}[DEBUG] Error during recorder shutdown: {e}{bcolors.ENDC}")
        
        if self.recorder_thread:
            self.recorder_thread.join(timeout=5)
            if self.recorder_thread.is_alive():
                print(f"{bcolors.WARNING}[DEBUG] Recorder thread did not terminate gracefully{bcolors.ENDC}")

# Global variables
global_args = None
recorder_manager: Optional[HybridRecorderManager] = None
prev_text = ""

# Define allowed methods and parameters for security
allowed_methods = [
    'set_microphone',
    'abort',
    'stop',
    'clear_audio_queue',
    'wakeup',
    'shutdown',
    'text',
    'reset_recorder_state',
]
allowed_parameters = [
    'language',
    'silero_sensitivity',
    'wake_word_activation_delay',
    'post_speech_silence_duration',
    'listen_start',
    'recording_stop_time',
    'last_transcription_bytes',
    'last_transcription_bytes_b64',
    'speech_end_silence_start',
    'is_recording',
    'use_wake_words',
]

# Queues and connections for control and data
control_connections: Set[websockets.WebSocketServerProtocol] = set()
data_connections: Set[websockets.WebSocketServerProtocol] = set()
control_queue = asyncio.Queue()
audio_queue = asyncio.Queue()

# Connection management with weak references to prevent memory leaks
connection_refs: Set[weakref.ref] = set()

def cleanup_dead_connections():
    """Clean up dead connections to prevent memory leaks"""
    global connection_refs
    dead_refs = set()
    for ref in connection_refs:
        if ref() is None:
            dead_refs.add(ref)
    connection_refs -= dead_refs

def preprocess_text(text):
    # Remove leading whitespaces
    text = text.lstrip()

    # Remove starting ellipses if present
    if text.startswith("..."):
        text = text[3:]

    if text.endswith("...'."):
        text = text[:-1]

    if text.endswith("...'"):
        text = text[:-1]

    # Remove any leading whitespaces again after ellipses removal
    text = text.lstrip()

    # Uppercase the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

def debug_print(message):
    if debug_logging:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        thread_name = threading.current_thread().name
        print(f"{Fore.CYAN}[DEBUG][{timestamp}][{thread_name}] {message}{Style.RESET_ALL}", file=sys.stderr)

def format_timestamp_ns(timestamp_ns: int) -> str:
    # Split into whole seconds and the nanosecond remainder
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # Convert seconds part into a datetime object (local time)
    dt = datetime.fromtimestamp(seconds)

    # Format the main time as HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")

    # For instance, if you want milliseconds, divide the remainder by 1e6 and format as 3-digit
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

def text_detected(text, loop):
    global prev_text

    print(f"{bcolors.OKCYAN}[DEBUG] text_detected called with: '{text}'{bcolors.ENDC}")

    text = preprocess_text(text)
    
    # Update metrics
    server_metrics.reset_activity()

    if silence_timing:
        def ends_with_ellipsis(text: str):
            if text.endswith("..."):
                return True
            if len(text) > 1 and text[:-1].endswith("..."):
                return True
            return False

        def sentence_end(text: str):
            sentence_end_marks = ['.', '!', '?', '。']
            if text and text[-1] in sentence_end_marks:
                return True
            return False

        if recorder_manager and recorder_manager.recorder:
            try:
                if ends_with_ellipsis(text):
                    recorder_manager.recorder.post_speech_silence_duration = global_args.mid_sentence_detection_pause
                elif sentence_end(text) and sentence_end(prev_text) and not ends_with_ellipsis(prev_text):
                    recorder_manager.recorder.post_speech_silence_duration = global_args.end_of_sentence_detection_pause
                else:
                    recorder_manager.recorder.post_speech_silence_duration = global_args.unknown_sentence_detection_pause
            except Exception as e:
                print(f"{bcolors.WARNING}[DEBUG] Error updating silence duration: {e}{bcolors.ENDC}")

    prev_text = text

    # Put the message in the audio queue to be sent to clients
    message = json.dumps({
        'type': 'realtime',
        'text': text
    })
    print(f"{bcolors.OKGREEN}[DEBUG] Queuing realtime message: {message}{bcolors.ENDC}")
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

    # Get current timestamp in HH:MM:SS.nnn format
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

    if extended_logging:
        print(f"  [{timestamp}] Realtime text: {bcolors.OKCYAN}{text}{bcolors.ENDC}\n", flush=True, end="")
    else:
        print(f"\r[{timestamp}] {bcolors.OKCYAN}{text}{bcolors.ENDC}", flush=True, end='')

def reset_recorder_state():
    """Reset recorder state to default values"""
    global prev_text
    if recorder_manager and recorder_manager.recorder:
        try:
            # Reset silence duration to default
            recorder_manager.recorder.post_speech_silence_duration = global_args.unknown_sentence_detection_pause
            # Don't clear audio queue or stop recording as it might interfere with ongoing processing
            # Only reset global variables
        except Exception as e:
            print(f"{bcolors.WARNING}[DEBUG] Error in reset_recorder_state: {e}{bcolors.ENDC}")
    # Reset global variables
    prev_text = ""
    print(f"{bcolors.OKGREEN}[DEBUG] Recorder state reset{bcolors.ENDC}")

def on_recording_start(loop):
    message = json.dumps({
        'type': 'recording_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_recording_stop(loop):
    message = json.dumps({
        'type': 'recording_stop'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_vad_detect_start(loop):
    message = json.dumps({
        'type': 'vad_detect_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_vad_detect_stop(loop):
    message = json.dumps({
        'type': 'vad_detect_stop'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detected(loop):
    message = json.dumps({
        'type': 'wakeword_detected'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detection_start(loop):
    message = json.dumps({
        'type': 'wakeword_detection_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detection_end(loop):
    message = json.dumps({
        'type': 'wakeword_detection_end'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_transcription_start(_audio_bytes, loop):
    bytes_b64 = base64.b64encode(_audio_bytes.tobytes()).decode('utf-8')
    message = json.dumps({
        'type': 'transcription_start',
        'audio_bytes_base64': bytes_b64
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_turn_detection_start(loop):
    print("&&& stt_server on_turn_detection_start")
    message = json.dumps({
        'type': 'start_turn_detection'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_turn_detection_stop(loop):
    print("&&& stt_server on_turn_detection_stop")
    message = json.dumps({
        'type': 'stop_turn_detection'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_realtime_transcription_update(text, loop):
    # Send real-time transcription updates to the client
    text = preprocess_text(text)
    message = json.dumps({
        'type': 'realtime_update',
        'text': text
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_recorded_chunk(chunk, loop):
    if send_recorded_chunk:
        bytes_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
        message = json.dumps({
            'type': 'recorded_chunk',
            'bytes': bytes_b64
        })
        asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def decode_and_resample(
        audio_data,
        original_sample_rate,
        target_sample_rate):

    # Decode 16-bit PCM data to numpy array
    if original_sample_rate == target_sample_rate:
        return audio_data

    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Calculate the number of samples after resampling
    num_original_samples = len(audio_np)
    num_target_samples = int(num_original_samples * target_sample_rate /
                                original_sample_rate)

    # Resample the audio
    resampled_audio = resample(audio_np, num_target_samples)

    return resampled_audio.astype(np.int16).tobytes()

# Define the server's arguments
def parse_arguments():
    global debug_logging, extended_logging, loglevel, writechunks, log_incoming_chunks, dynamic_silence_timing

    import argparse
    parser = argparse.ArgumentParser(description='Start the Speech-to-Text (STT) server with various configuration options.')

    parser.add_argument('-m', '--model', type=str, default='large-v2',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, default='tiny',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is tiny.en.')
    
    parser.add_argument('-l', '--lang', '--language', type=str, default='en',
                help='Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')

    parser.add_argument('-i', '--input-device', '--input-device-index', type=int, default=1,
                    help='Index of the audio input device to use. Use this option to specify a particular microphone or audio input device based on your system. Default is 1.')

    parser.add_argument('-c', '--control', '--control_port', type=int, default=8011,
                        help='The port number used for the control WebSocket connection. Control connections are used to send and receive commands to the server. Default is port 8011.')

    parser.add_argument('-d', '--data', '--data_port', type=int, default=8012,
                        help='The port number used for the data WebSocket connection. Data connections are used to send audio data and receive transcription updates in real time. Default is port 8012.')

    parser.add_argument('-w', '--wake_words', type=str, default="",
                        help='Specify the wake word(s) that will trigger the server to start listening. For example, setting this to "Jarvis" will make the system start transcribing when it detects the wake word "Jarvis". Default is "Jarvis".')

    parser.add_argument('-D', '--debug', action='store_true', help='Enable debug logging for detailed server operations')

    parser.add_argument('--debug_websockets', action='store_true', help='Enable debug logging for detailed server websocket operations')

    parser.add_argument('-W', '--write', metavar='FILE', help='Save received audio to a WAV file')
    
    parser.add_argument('-b', '--batch', '--batch_size', type=int, default=16, help='Batch size for inference. This parameter controls the number of audio chunks processed in parallel during transcription. Default is 16.')

    parser.add_argument('--root', '--download_root', type=str,default=None, help='Specifies the root path where the Whisper models are downloaded to. Default is None.')

    parser.add_argument('-s', '--silence_timing', action='store_true', default=True,
                    help='Enable dynamic adjustment of silence duration for sentence detection. Adjusts post-speech silence duration based on detected sentence structure and punctuation. Default is False.')

    parser.add_argument('--init_realtime_after_seconds', type=float, default=0.2,
                        help='The initial waiting time in seconds before real-time transcription starts. This delay helps prevent false positives at the beginning of a session. Default is 0.2 seconds.')  
    
    parser.add_argument('--realtime_batch_size', type=int, default=16,
                        help='Batch size for the real-time transcription model. This parameter controls the number of audio chunks processed in parallel during real-time transcription. Default is 16.')
    
    parser.add_argument('--initial_prompt_realtime', type=str, default="", help='Initial prompt that guides the real-time transcription model to produce transcriptions in a particular style or format.')

    parser.add_argument('--silero_sensitivity', type=float, default=0.05,
                        help='Sensitivity level for Silero Voice Activity Detection (VAD), with a range from 0 to 1. Lower values make the model less sensitive, useful for noisy environments. Default is 0.05.')

    parser.add_argument('--silero_use_onnx', action='store_true', default=False,
                        help='Enable ONNX version of Silero model for faster performance with lower resource usage. Default is False.')

    parser.add_argument('--webrtc_sensitivity', type=int, default=3,
                        help='Sensitivity level for WebRTC Voice Activity Detection (VAD), with a range from 0 to 3. Higher values make the model less sensitive, useful for cleaner environments. Default is 3.')

    parser.add_argument('--min_length_of_recording', type=float, default=1.1,
                        help='Minimum duration of valid recordings in seconds. This prevents very short recordings from being processed, which could be caused by noise or accidental sounds. Default is 1.1 seconds.')

    parser.add_argument('--min_gap_between_recordings', type=float, default=0,
                        help='Minimum time (in seconds) between consecutive recordings. Setting this helps avoid overlapping recordings when there’s a brief silence between them. Default is 0 seconds.')

    parser.add_argument('--enable_realtime_transcription', action='store_true', default=True,
                        help='Enable continuous real-time transcription of audio as it is received. When enabled, transcriptions are sent in near real-time. Default is True.')

    parser.add_argument('--realtime_processing_pause', type=float, default=0.02,
                        help='Time interval (in seconds) between processing audio chunks for real-time transcription. Lower values increase responsiveness but may put more load on the CPU. Default is 0.02 seconds.')

    parser.add_argument('--silero_deactivity_detection', action='store_true', default=True,
                        help='Use the Silero model for end-of-speech detection. This option can provide more robust silence detection in noisy environments, though it consumes more GPU resources. Default is True.')

    parser.add_argument('--early_transcription_on_silence', type=float, default=0.2,
                        help='Start transcription after the specified seconds of silence. This is useful when you want to trigger transcription mid-speech when there is a brief pause. Should be lower than post_speech_silence_duration. Set to 0 to disable. Default is 0.2 seconds.')

    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for the main transcription model. Larger values may improve transcription accuracy but increase the processing time. Default is 5.')

    parser.add_argument('--beam_size_realtime', type=int, default=3,
                        help='Beam size for the real-time transcription model. A smaller beam size allows for faster real-time processing but may reduce accuracy. Default is 3.')

    parser.add_argument('--initial_prompt', type=str,
                        default="Incomplete thoughts should end with '...'. Examples of complete thoughts: 'The sky is blue.' 'She walked home.' Examples of incomplete thoughts: 'When the sky...' 'Because he...'",
                        help='Initial prompt that guides the transcription model to produce transcriptions in a particular style or format. The default provides instructions for handling sentence completions and ellipsis usage.')

    parser.add_argument('--end_of_sentence_detection_pause', type=float, default=0.45,
                        help='The duration of silence (in seconds) that the model should interpret as the end of a sentence. This helps the system detect when to finalize the transcription of a sentence. Default is 0.45 seconds.')

    parser.add_argument('--unknown_sentence_detection_pause', type=float, default=0.7,
                        help='The duration of pause (in seconds) that the model should interpret as an incomplete or unknown sentence. This is useful for identifying when a sentence is trailing off or unfinished. Default is 0.7 seconds.')

    parser.add_argument('--mid_sentence_detection_pause', type=float, default=2.0,
                        help='The duration of pause (in seconds) that the model should interpret as a mid-sentence break. Longer pauses can indicate a pause in speech but not necessarily the end of a sentence. Default is 2.0 seconds.')

    parser.add_argument('--wake_words_sensitivity', type=float, default=0.5,
                        help='Sensitivity level for wake word detection, with a range from 0 (most sensitive) to 1 (least sensitive). Adjust this value based on your environment to ensure reliable wake word detection. Default is 0.5.')

    parser.add_argument('--wake_word_timeout', type=float, default=5.0,
                        help='Maximum time in seconds that the system will wait for a wake word before timing out. After this timeout, the system stops listening for wake words until reactivated. Default is 5.0 seconds.')

    parser.add_argument('--wake_word_activation_delay', type=float, default=0,
                        help='The delay in seconds before the wake word detection is activated after the system starts listening. This prevents false positives during the start of a session. Default is 0 seconds.')

    parser.add_argument('--wakeword_backend', type=str, default='none',
                        help='The backend used for wake word detection. You can specify different backends such as "default" or any custom implementations depending on your setup. Default is "pvporcupine".')

    parser.add_argument('--openwakeword_model_paths', type=str, nargs='*',
                        help='A list of file paths to OpenWakeWord models. This is useful if you are using OpenWakeWord for wake word detection and need to specify custom models.')

    parser.add_argument('--openwakeword_inference_framework', type=str, default='tensorflow',
                        help='The inference framework to use for OpenWakeWord models. Supported frameworks could include "tensorflow", "pytorch", etc. Default is "tensorflow".')

    parser.add_argument('--wake_word_buffer_duration', type=float, default=1.0,
                        help='Duration of the buffer in seconds for wake word detection. This sets how long the system will store the audio before and after detecting the wake word. Default is 1.0 seconds.')

    parser.add_argument('--use_main_model_for_realtime', action='store_true',
                        help='Enable this option if you want to use the main model for real-time transcription, instead of the smaller, faster real-time model. Using the main model may provide better accuracy but at the cost of higher processing time.')

    parser.add_argument('--use_extended_logging', action='store_true',
                        help='Writes extensive log messages for the recording worker, that processes the audio chunks.')

    parser.add_argument('--compute_type', type=str, default='default',
                        help='Type of computation to use. See https://opennmt.net/CTranslate2/quantization.html')

    parser.add_argument('--gpu_device_index', type=int, default=0,
                        help='Index of the GPU device to use. Default is None.')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for model to use. Can either be "cuda" or "cpu". Default is cuda.')
    
    parser.add_argument('--handle_buffer_overflow', action='store_true',
                        help='Handle buffer overflow during transcription. Default is False.')

    parser.add_argument('--suppress_tokens', type=int, default=[-1], nargs='*', help='Suppress tokens during transcription. Default is [-1].')

    parser.add_argument('--allowed_latency_limit', type=int, default=100,
                        help='Maximal amount of chunks that can be unprocessed in queue before discarding chunks.. Default is 100.')

    parser.add_argument('--faster_whisper_vad_filter', action='store_true',
                        help='Enable VAD filter for Faster Whisper. Default is False.')

    parser.add_argument('--logchunks', action='store_true', help='Enable logging of incoming audio chunks (periods)')

    # Parse arguments
    args = parser.parse_args()

    debug_logging = args.debug
    extended_logging = args.use_extended_logging
    writechunks = args.write
    log_incoming_chunks = args.logchunks
    dynamic_silence_timing = args.silence_timing


    ws_logger = logging.getLogger('websockets')
    if args.debug_websockets:
        # If app debug is on, let websockets be verbose too
        ws_logger.setLevel(logging.DEBUG)
        # Ensure it uses the handler configured by basicConfig
        ws_logger.propagate = False # Prevent duplicate messages if it also propagates to root
    else:
        # If app debug is off, silence websockets below WARNING
        ws_logger.setLevel(logging.WARNING)
        ws_logger.propagate = True # Allow WARNING/ERROR messages to reach root logger's handler

    # Replace escaped newlines with actual newlines in initial_prompt
    if args.initial_prompt:
        args.initial_prompt = args.initial_prompt.replace("\\n", "\n")

    if args.initial_prompt_realtime:
        args.initial_prompt_realtime = args.initial_prompt_realtime.replace("\\n", "\n")

    return args

async def control_handler(websocket):
    debug_print(f"New control connection from {websocket.remote_address}")
    print(f"{bcolors.OKGREEN}Control client connected{bcolors.ENDC}")
    global recorder_manager, server_metrics
    
    # Track connection metrics
    server_metrics.active_connections += 1
    server_metrics.total_connections += 1
    
    control_connections.add(websocket)
    connection_refs.add(weakref.ref(websocket))
    
    # Reset recorder state for new control connection
    reset_recorder_state()
    
    try:
        async for message in websocket:
            debug_print(f"Received control message: {message[:200]}...")
            if not recorder_manager.ready_event.is_set():
                print(f"{bcolors.WARNING}Recorder not ready{bcolors.ENDC}")
                continue
            if isinstance(message, str):
                # Handle text message (command)
                try:
                    command_data = json.loads(message)
                    command = command_data.get("command")
                    if command == "set_parameter":
                        parameter = command_data.get("parameter")
                        value = command_data.get("value")
                        if parameter in allowed_parameters and hasattr(recorder_manager.recorder, parameter):
                            try:
                                setattr(recorder_manager.recorder, parameter, value)
                            except Exception as e:
                                print(f"{bcolors.WARNING}[DEBUG] Error setting parameter {parameter}: {e}{bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Error setting parameter {parameter}: {e}"}))
                                continue
                            # Format the value for output
                            if isinstance(value, float):
                                value_formatted = f"{value:.2f}"
                            else:
                                value_formatted = value
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            if extended_logging:
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Set recorder.{parameter} to: {bcolors.OKBLUE}{value_formatted}{bcolors.ENDC}")
                            # Optionally send a response back to the client
                            await websocket.send(json.dumps({"status": "success", "message": f"Parameter {parameter} set to {value}"}))
                        else:
                            if not parameter in allowed_parameters:
                                print(f"{bcolors.WARNING}Parameter {parameter} is not allowed (set_parameter){bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} is not allowed (set_parameter)"}))
                            else:
                                print(f"{bcolors.WARNING}Parameter {parameter} does not exist (set_parameter){bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} does not exist (set_parameter)"}))

                    elif command == "get_parameter":
                        parameter = command_data.get("parameter")
                        request_id = command_data.get("request_id")  # Get the request_id from the command data
                        if parameter in allowed_parameters and hasattr(recorder_manager.recorder, parameter):
                            try:
                                value = getattr(recorder_manager.recorder, parameter)
                            except Exception as e:
                                print(f"{bcolors.WARNING}[DEBUG] Error getting parameter {parameter}: {e}{bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Error getting parameter {parameter}: {e}"}))
                                continue
                            if isinstance(value, float):
                                value_formatted = f"{value:.2f}"
                            else:
                                value_formatted = f"{value}"

                            value_truncated = value_formatted[:39] + "…" if len(value_formatted) > 40 else value_formatted

                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            if extended_logging:
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Get recorder.{parameter}: {bcolors.OKBLUE}{value_truncated}{bcolors.ENDC}")
                            response = {"status": "success", "parameter": parameter, "value": value}
                            if request_id is not None:
                                response["request_id"] = request_id
                            await websocket.send(json.dumps(response))
                        else:
                            if not parameter in allowed_parameters:
                                print(f"{bcolors.WARNING}Parameter {parameter} is not allowed (get_parameter){bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} is not allowed (get_parameter)"}))
                            else:
                                print(f"{bcolors.WARNING}Parameter {parameter} does not exist (get_parameter){bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} does not exist (get_parameter)"}))
                    elif command == "call_method":
                        method_name = command_data.get("method")
                        if method_name in allowed_methods:
                            method = getattr(recorder_manager.recorder, method_name, None)
                            if method and callable(method):
                                args = command_data.get("args", [])
                                kwargs = command_data.get("kwargs", {})
                                try:
                                    method(*args, **kwargs)
                                except Exception as e:
                                    print(f"{bcolors.WARNING}[DEBUG] Error calling method {method_name}: {e}{bcolors.ENDC}")
                                    await websocket.send(json.dumps({"status": "error", "message": f"Error calling method {method_name}: {e}"}))
                                    continue
                                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Called method recorder.{bcolors.OKBLUE}{method_name}{bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "success", "message": f"Method {method_name} called"}))
                            else:
                                print(f"{bcolors.WARNING}Recorder does not have method {method_name}{bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Recorder does not have method {method_name}"}))
                        else:
                            print(f"{bcolors.WARNING}Method {method_name} is not allowed{bcolors.ENDC}")
                            await websocket.send(json.dumps({"status": "error", "message": f"Method {method_name} is not allowed"}))
                    else:
                        print(f"{bcolors.WARNING}Unknown command: {command}{bcolors.ENDC}")
                        await websocket.send(json.dumps({"status": "error", "message": f"Unknown command {command}"}))
                except json.JSONDecodeError:
                    print(f"{bcolors.WARNING}Received invalid JSON command{bcolors.ENDC}")
                    await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON command"}))
            else:
                print(f"{bcolors.WARNING}Received unknown message type on control connection{bcolors.ENDC}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"{bcolors.WARNING}Control client disconnected: {e}{bcolors.ENDC}")
    finally:
        control_connections.remove(websocket)
        server_metrics.active_connections -= 1
        cleanup_dead_connections()
        # Reset recorder state when control connection closes
        reset_recorder_state()

async def data_handler(websocket):
    global writechunks, wav_file, server_metrics
    print(f"{bcolors.OKGREEN}Data client connected{bcolors.ENDC}")
    
    # Track connection metrics
    server_metrics.active_connections += 1
    server_metrics.total_connections += 1
    
    data_connections.add(websocket)
    connection_refs.add(weakref.ref(websocket))
    print(f"{bcolors.OKCYAN}[DEBUG] Total data connections: {len(data_connections)}{bcolors.ENDC}")
    
    # Reset recorder state for new connection
    reset_recorder_state()
    
    try:
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                if extended_logging:
                    debug_print(f"Received audio chunk (size: {len(message)} bytes)")
                elif log_incoming_chunks:
                    print(".", end='', flush=True)
                # Handle binary message (audio data)
                metadata_length = int.from_bytes(message[:4], byteorder='little')
                metadata_json = message[4:4+metadata_length].decode('utf-8')
                metadata = json.loads(metadata_json)
                sample_rate = metadata['sampleRate']

                if 'server_sent_to_stt' in metadata:
                    stt_received_ns = time.time_ns()
                    metadata["stt_received"] = stt_received_ns
                    metadata["stt_received_formatted"] = format_timestamp_ns(stt_received_ns)
                    print(f"Server received audio chunk of length {len(message)} bytes, metadata: {metadata}")

                if extended_logging:
                    debug_print(f"Processing audio chunk with sample rate {sample_rate}")
                chunk = message[4+metadata_length:]

                if writechunks:
                    if not wav_file:
                        wav_file = wave.open(writechunks, 'wb')
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
                        wav_file.setframerate(sample_rate)

                    wav_file.writeframes(chunk)

                if sample_rate != 16000:
                    resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
                    if extended_logging:
                        debug_print(f"Resampled chunk size: {len(resampled_chunk)} bytes")
                    await recorder_manager.process_audio(resampled_chunk)
                else:
                    await recorder_manager.process_audio(chunk)
                
                # Update heartbeat timer when audio is fed to recorder
                if 'last_activity_time' in globals():
                    globals()['last_activity_time'] = time.time()
            else:
                print(f"{bcolors.WARNING}Received non-binary message on data connection{bcolors.ENDC}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"{bcolors.WARNING}Data client disconnected: {e}{bcolors.ENDC}")
    finally:
        data_connections.discard(websocket)
        server_metrics.active_connections -= 1
        cleanup_dead_connections()
        print(f"{bcolors.OKCYAN}[DEBUG] Total data connections after disconnect: {len(data_connections)}{bcolors.ENDC}")
        # Reset recorder state when connection closes
        reset_recorder_state()

async def broadcast_audio_messages():
    while True:
        message = await audio_queue.get()
        print(f"{bcolors.OKCYAN}[DEBUG] Broadcasting message to {len(data_connections)} connections{bcolors.ENDC}")
        for conn in list(data_connections):
            try:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                if extended_logging:
                    print(f"  [{timestamp}] Sending message: {bcolors.OKBLUE}{message}{bcolors.ENDC}\n", flush=True, end="")
                await conn.send(message)
                print(f"{bcolors.OKGREEN}[DEBUG] Successfully sent message to connection{bcolors.ENDC}")
            except websockets.exceptions.ConnectionClosed as e:
                print(f"{bcolors.WARNING}[DEBUG] Connection closed while sending: {e}{bcolors.ENDC}")
                data_connections.discard(conn)
            except Exception as e:
                print(f"{bcolors.FAIL}[DEBUG] Error sending message: {e}{bcolors.ENDC}")
                # Don't remove connection for general exceptions, only for ConnectionClosed

# Production-ready HTTP health check server with metrics
async def health_check_handler(request):
    """Production health check endpoint with detailed status"""
    global recorder_health, server_metrics
    
    # Check if recorder is healthy
    is_healthy = (
        recorder_health.state in [RecorderState.READY, RecorderState.PROCESSING] and
        server_metrics.get_uptime() > 0
    )
    
    status_code = 200 if is_healthy else 503
    
    health_data = {
        "status": "healthy" if is_healthy else "unhealthy",
        "recorder_state": recorder_health.state.value,
        "uptime_seconds": round(server_metrics.get_uptime(), 2),
        "active_connections": server_metrics.active_connections,
        "total_connections": server_metrics.total_connections,
        "connections_per_minute": round(server_metrics.get_connections_per_minute(), 2),
        "audio_chunks_processed": server_metrics.audio_chunks_processed,
        "transcription_errors": server_metrics.transcription_errors,
        "recorder_errors": server_metrics.recorder_errors,
        "last_activity_seconds": round(time.time() - server_metrics.last_activity, 2),
        "consecutive_errors": recorder_health.consecutive_errors,
        "recovery_attempts": recorder_health.recovery_attempts
    }
    
    return web.json_response(health_data, status=status_code)

async def metrics_handler(request):
    """Detailed metrics endpoint for monitoring"""
    global recorder_health, server_metrics
    
    metrics_data = {
        "server": {
            "uptime_seconds": round(server_metrics.get_uptime(), 2),
            "start_time": server_metrics.start_time,
            "last_activity": server_metrics.last_activity
        },
        "connections": {
            "active": server_metrics.active_connections,
            "total": server_metrics.total_connections,
            "rate_per_minute": round(server_metrics.get_connections_per_minute(), 2)
        },
        "processing": {
            "audio_chunks_processed": server_metrics.audio_chunks_processed,
            "transcription_errors": server_metrics.transcription_errors,
            "recorder_errors": server_metrics.recorder_errors
        },
        "recorder": {
            "state": recorder_health.state.value,
            "last_successful_processing": recorder_health.last_successful_processing,
            "consecutive_errors": recorder_health.consecutive_errors,
            "recovery_attempts": recorder_health.recovery_attempts,
            "can_recover": recorder_health.can_recover()
        }
    }
    
    return web.json_response(metrics_data)

async def start_health_check_server(port=8080):
    """Start a production-ready HTTP health check server on the specified port"""
    app = web.Application()
    app.router.add_get("/health", health_check_handler)
    app.router.add_get("/metrics", metrics_handler)
    app.router.add_get("/", health_check_handler)  # Also respond to root path
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    
    print(f"{bcolors.OKGREEN}Production health check server started on {bcolors.OKBLUE}http://0.0.0.0:{port}/health{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Metrics endpoint available at {bcolors.OKBLUE}http://0.0.0.0:{port}/metrics{bcolors.ENDC}")
    return runner

async def production_monitoring():
    """Production monitoring task that runs without blocking"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Log production metrics
            uptime = server_metrics.get_uptime()
            connections_per_min = server_metrics.get_connections_per_minute()
            
            print(f"{bcolors.OKCYAN}[PRODUCTION] Uptime: {uptime:.1f}s, "
                  f"Active: {server_metrics.active_connections}, "
                  f"Total: {server_metrics.total_connections}, "
                  f"Rate: {connections_per_min:.1f}/min, "
                  f"Audio chunks: {server_metrics.audio_chunks_processed}, "
                  f"Errors: {server_metrics.transcription_errors + server_metrics.recorder_errors}, "
                  f"Recorder state: {recorder_health.state.value}{bcolors.ENDC}")
            
            # Check for potential issues
            if server_metrics.transcription_errors + server_metrics.recorder_errors > 100:
                print(f"{bcolors.WARNING}[PRODUCTION] High error rate detected!{bcolors.ENDC}")
            
            if recorder_health.state == RecorderState.ERROR:
                print(f"{bcolors.WARNING}[PRODUCTION] Recorder in error state!{bcolors.ENDC}")
            
            # Clean up dead connections periodically
            cleanup_dead_connections()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"{bcolors.WARNING}[PRODUCTION] Monitoring error: {e}{bcolors.ENDC}")

# Helper function to create event loop bound closures for callbacks
def make_callback(loop, callback):
    def inner_callback(*args, **kwargs):
        callback(*args, **kwargs, loop=loop)
    return inner_callback

async def main_async():            
    global recorder_manager, global_args
    args = parse_arguments()
    global_args = args

    # Get the event loop here and pass it to the recorder thread
    loop = asyncio.get_event_loop()

    recorder_manager = HybridRecorderManager(
        {
            'model': args.model,
            'download_root': args.root,
            'realtime_model_type': args.rt_model,
            'language': args.lang,
            'batch_size': args.batch,
            'init_realtime_after_seconds': args.init_realtime_after_seconds,
            'realtime_batch_size': args.realtime_batch_size,
            'initial_prompt_realtime': args.initial_prompt_realtime,
            'input_device_index': args.input_device,
            'silero_sensitivity': args.silero_sensitivity,
            'silero_use_onnx': args.silero_use_onnx,
            'webrtc_sensitivity': args.webrtc_sensitivity,
            'post_speech_silence_duration': args.unknown_sentence_detection_pause,
            'min_length_of_recording': args.min_length_of_recording,
            'min_gap_between_recordings': args.min_gap_between_recordings,
            'enable_realtime_transcription': args.enable_realtime_transcription,
            'realtime_processing_pause': args.realtime_processing_pause,
            'silero_deactivity_detection': args.silero_deactivity_detection,
            'early_transcription_on_silence': args.early_transcription_on_silence,
            'beam_size': args.beam_size,
            'beam_size_realtime': args.beam_size_realtime,
            'initial_prompt': args.initial_prompt,
            'wake_words': args.wake_words,
            'wake_words_sensitivity': args.wake_words_sensitivity,
            'wake_word_timeout': args.wake_word_timeout,
            'wake_word_activation_delay': args.wake_word_activation_delay,
            'wakeword_backend': args.wakeword_backend,
            'openwakeword_model_paths': args.openwakeword_model_paths,
            'openwakeword_inference_framework': args.openwakeword_inference_framework,
            'wake_word_buffer_duration': args.wake_word_buffer_duration,
            'use_main_model_for_realtime': args.use_main_model_for_realtime,
            'spinner': False,
            'use_microphone': False,

            'on_realtime_transcription_update': make_callback(loop, text_detected),
            'on_recording_start': make_callback(loop, on_recording_start),
            'on_recording_stop': make_callback(loop, on_recording_stop),
            'on_vad_detect_start': make_callback(loop, on_vad_detect_start),
            'on_vad_detect_stop': make_callback(loop, on_vad_detect_stop),
            'on_wakeword_detected': make_callback(loop, on_wakeword_detected),
            'on_wakeword_detection_start': make_callback(loop, on_wakeword_detection_start),
            'on_wakeword_detection_end': make_callback(loop, on_wakeword_detection_end),
            'on_transcription_start': make_callback(loop, on_transcription_start),
            'on_turn_detection_start': make_callback(loop, on_turn_detection_start),
            'on_turn_detection_stop': make_callback(loop, on_turn_detection_stop),

            'on_recorded_chunk': make_callback(loop, on_recorded_chunk),
            'no_log_file': True,  # Disable logging to file
            'use_extended_logging': args.use_extended_logging,
            'level': loglevel,
            'compute_type': args.compute_type,
            'gpu_device_index': args.gpu_device_index,
            'device': args.device,
            'handle_buffer_overflow': args.handle_buffer_overflow,
            'suppress_tokens': args.suppress_tokens,
            'allowed_latency_limit': args.allowed_latency_limit,
            'faster_whisper_vad_filter': args.faster_whisper_vad_filter,
        },
        loop
    )

    try:
        # Attempt to start control and data servers with proper error handling
        try:
            control_server = await serve(
                control_handler, 
                "0.0.0.0", 
                args.control
            )
            data_server = await serve(
                data_handler, 
                "0.0.0.0", 
                args.data
            )
            print(f"{bcolors.OKGREEN}Control server started on {bcolors.OKBLUE}ws://0.0.0.0:{args.control}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}Data server started on {bcolors.OKBLUE}ws://0.0.0.0:{args.data}{bcolors.ENDC}")
        except Exception as e:
            print(f"{bcolors.FAIL}Error starting WebSocket servers: {e}{bcolors.ENDC}")
            raise

        # Start HTTP health check server
        try:
            health_check_runner = await start_health_check_server(port=8080)
        except Exception as e:
            print(f"{bcolors.FAIL}Error starting health check server: {e}{bcolors.ENDC}")
            raise

        # Start the broadcast and recorder threads
        broadcast_task = asyncio.create_task(broadcast_audio_messages())

        # Initialize recorder and start processing
        await recorder_manager.initialize()
        # The recorder_manager.start_processing() call is now handled by the recorder_thread
        
        # Start production monitoring task
        monitoring_task = asyncio.create_task(production_monitoring())

        print(f"{bcolors.OKGREEN}Production server started. Press Ctrl+C to stop the server.{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Monitoring: {bcolors.OKBLUE}http://0.0.0.0:8080/health{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Metrics: {bcolors.OKBLUE}http://0.0.0.0:8080/metrics{bcolors.ENDC}")

        # Run server tasks - wait for broadcast task to complete or server shutdown
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
    except OSError as e:
        print(f"{bcolors.FAIL}Error: Could not start server on specified ports. It’s possible another instance of the server is already running, or the ports are being used by another application.{bcolors.ENDC}")
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Server interrupted by user, shutting down...{bcolors.ENDC}")
    finally:
        # Shutdown procedures for recorder and server threads
        await shutdown_procedure()
        print(f"{bcolors.OKGREEN}Server shutdown complete.{bcolors.ENDC}")

async def shutdown_procedure():
    global recorder_manager, health_check_runner
    
    print(f"{bcolors.OKGREEN}Starting graceful shutdown...{bcolors.ENDC}")
    
    # Close WebSocket servers first
    if 'control_server' in globals() and control_server:
        control_server.close()
        await control_server.wait_closed()
        print(f"{bcolors.OKGREEN}Control server closed{bcolors.ENDC}")
    
    if 'data_server' in globals() and data_server:
        data_server.close()
        await data_server.wait_closed()
        print(f"{bcolors.OKGREEN}Data server closed{bcolors.ENDC}")
    
    # Close health check server if it exists
    if 'health_check_runner' in globals() and health_check_runner:
        await health_check_runner.cleanup()
        print(f"{bcolors.OKGREEN}Health check server closed{bcolors.ENDC}")
    
    # Shutdown recorder manager gracefully
    if recorder_manager:
        await recorder_manager.shutdown()
        print(f"{bcolors.OKGREEN}Recorder manager shut down{bcolors.ENDC}")

    # Close WAV file if it's open
    if wav_file:
        try:
            wav_file.close()
            print(f"{bcolors.OKGREEN}WAV file closed{bcolors.ENDC}")
        except Exception as e:
            debug_print(f"Error closing WAV file: {e}")

    # Print final metrics
    print(f"{bcolors.OKCYAN}[FINAL METRICS] Uptime: {server_metrics.get_uptime():.1f}s, "
          f"Total connections: {server_metrics.total_connections}, "
          f"Audio chunks processed: {server_metrics.audio_chunks_processed}, "
          f"Total errors: {server_metrics.transcription_errors + server_metrics.recorder_errors}{bcolors.ENDC}")

    # Cancel all remaining tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print(f"{bcolors.OKGREEN}Graceful shutdown complete.{bcolors.ENDC}")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        # Capture any final KeyboardInterrupt to prevent it from showing up in logs
        print(f"{bcolors.WARNING}Server interrupted by user.{bcolors.ENDC}")
        exit(0)

if __name__ == '__main__':
    main()
