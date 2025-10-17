from typing import Optional
from ..types import ViewerMode, LOCAL_CLIENT
from abc import ABC, abstractmethod

class Widget(ABC):
    id = 0

    def __init__(self, mode: ViewerMode):
        self.mode = mode
        self.widget_id = Widget.id
        Widget.id += 1

        if mode & LOCAL_CLIENT:
            self.import_client_modules()

    def setup(self):
        """
        Perform any setup actions required after OpenGL/GLFW/ImGUI is initialized. 
        This function won't be called when application is running in headless mode.
        """

    def destroy(self):
        """
        Destroy any resources created manually in the 'setup' function.
        This function won't be called when application is running in headless mode.
        """

    def server_send(self) -> tuple[Optional[bytes], Optional[dict]]:
        """
        Send widget state to the client.

        Returns:
            binary (bytes): Any binary data to be sent to the client.
            text (dict): Any text data to be sent to the client.
        """
        return None, None
    
    def server_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """
        Receive widget state from the client and update it.

        Args:
            binary (bytes): Any binary data received from the client.
            text (dict): Any text data sent received from the client
        """
    
    def client_send(self) -> tuple[Optional[bytes], Optional[dict]]:
        """
        Send widget state to the server.

        Returns:
            binary (bytes): Any binary data to be sent to the server.
            text (dict): Any text data to be sent to the server.
        """
        return None, None

    def client_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """
        Send widget state to the server.

        Args:
            binary (bytes): Any binary data to be received from the server.
            text (dict): Any text data to be received from the server.
        """

    def import_client_modules(self):
        """
        Import client specific modules here. We want the viewer to run without 
        needing to install `imgui_bundle` on the server. The modules imported
        here can only be used in `show_gui` and `client*` methods. Don't forget
        to declare the variables as `global` to  ensure that they are globally
        accesible.
        """

    @abstractmethod
    def show_gui(self):
        pass