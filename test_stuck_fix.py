#!/usr/bin/env python3
"""
Test script to verify the stuck recorder fixes
"""

import asyncio
import websockets
import json
import time
import numpy as np

async def test_stuck_recorder():
    """Test the STT server to ensure it doesn't get stuck"""
    
    # Connect to the STT server
    control_uri = "ws://localhost:8765"
    data_uri = "ws://localhost:8765"
    
    try:
        # Connect to control websocket
        async with websockets.connect(control_uri) as control_ws:
            print("Connected to STT server control WebSocket")
            
            # Connect to data websocket
            async with websockets.connect(data_uri) as data_ws:
                print("Connected to STT server data WebSocket")
                
                # Send control message to start transcription
                control_message = {
                    "type": "start_transcription",
                    "sample_rate": 16000,
                    "channels": 1
                }
                await control_ws.send(json.dumps(control_message))
                print("Sent start transcription message")
                
                # Listen for messages in background
                async def listen_for_messages():
                    try:
                        while True:
                            message = await control_ws.recv()
                            print(f"Received message: {message}")
                    except websockets.exceptions.ConnectionClosed:
                        print("Control WebSocket connection closed")
                    except Exception as e:
                        print(f"Error receiving message: {e}")
                
                # Start listening task
                listen_task = asyncio.create_task(listen_for_messages())
                
                # Send some test audio data (silence)
                audio_data = np.zeros(1600, dtype=np.int16).tobytes()
                
                # Send audio in chunks to simulate speaking
                print("Sending audio chunks...")
                for i in range(20):
                    await data_ws.send(audio_data)
                    print(f"Sent audio chunk {i+1}")
                    await asyncio.sleep(0.1)
                
                # Wait a bit for processing
                await asyncio.sleep(2)
                
                # Send stop message
                stop_message = {"type": "stop_transcription"}
                await control_ws.send(json.dumps(stop_message))
                print("Sent stop transcription message")
                
                # Cancel listening task
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
                
    except Exception as e:
        print(f"Error: {e}")

async def test_health_check():
    """Test the health check endpoint"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/health') as response:
                health_data = await response.json()
                print(f"Health check response: {health_data}")
                
            async with session.get('http://localhost:8080/metrics') as response:
                metrics_data = await response.json()
                print(f"Metrics response: {metrics_data}")
                
    except Exception as e:
        print(f"Health check error: {e}")

if __name__ == "__main__":
    print("Testing STT server stuck recorder fixes...")
    
    print("\n1. Testing health endpoints...")
    asyncio.run(test_health_check())
    
    print("\n2. Testing WebSocket communication...")
    asyncio.run(test_stuck_recorder())
    
    print("Test completed") 