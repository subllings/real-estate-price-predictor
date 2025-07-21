from azure.servicebus import ServiceBusClient, ServiceBusReceiver

class TrainingJobServiceBusListener:
    def __init__(self):
        self.servicebus_client = ServiceBusClient.from_connection_string(conn_str)
    
    def listen_for_commands(self):
        with self.servicebus_client:
            receiver = self.servicebus_client.get_queue_receiver(queue_name="training-commands")
            
            with receiver:
                for msg in receiver:
                    command = json.loads(str(msg))
                    self.process_training_command(command)
                    receiver.complete_message(msg)
