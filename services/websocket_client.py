import asyncio
import websockets
import json

class TrainingJobWebSocketClient:
    async def connect_and_listen(self):
        uri = "ws://your-api-server:8001/training-commands"
        
        async with websockets.connect(uri) as websocket:
            print("🔗 Connected to training command server")
            
            async for message in websocket:
                command = json.loads(message)
                
                if command["action"] == "start_training":
                    await self.start_training_job(command["config"])
                elif command["action"] == "stop_training":
                    await self.stop_training_job(command["job_id"])
