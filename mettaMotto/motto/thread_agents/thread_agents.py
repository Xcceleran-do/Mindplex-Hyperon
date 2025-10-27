import datetime
import time
from motto.agents import DialogAgent, MettaAgent
from hyperon.ext import register_atoms
from hyperon.exts.agents.agent_base import StreamMethod
from queue import Queue
import threading
from motto.utils import *
import logging

system_event_description = {"speechstart": "[Event] User started speaking",
                            "speechcont": "[Event] User continues speaking", "speech": "[Event] User said sentence"}


class AgentArgs:
    def __init__(self, message, functions=[], additional_info=None, language=None, user_id=None):
        self.message = get_grounded_atom_value(message)
        self.additional_info = get_grounded_atom_value(additional_info)
        self.functions = get_grounded_atom_value(functions)
        self.language = language
        self.user_id = user_id


class ListeningAgent(DialogAgent):
    stop_message = "_stop_"

    # this method will be called via start in separate thread
    def __init__(self, path=None, atoms={}, include_paths=None, code=None, event_bus=None):
        self.log = logging.getLogger(__name__ + '.' + type(self).__name__)
        self.cancel_processing_var = False
        self.interrupt_processing_var = False
        self.processing = False

        atoms['handle-speechstart'] = OperationAtom('queue-subscription', self.handle_speechstart, unwrap=False)
        atoms['handle-speechcont'] = OperationAtom('handle-speechcont', self.handle_speechcont, unwrap=False)
        atoms['handle-speech'] = OperationAtom('handle-speech', self.handle_speech, unwrap=False)

        if isinstance(atoms, GroundedAtom):
            atoms = atoms.get_object().content
        super().__init__(path, atoms, include_paths, code, event_bus=event_bus)
        self.lock = threading.RLock()
        self.messages = Queue()
        self.speech_start = None

    def _postproc(self, response):
        # do not need to save history here so the method from MettaAgent is used
        result = MettaAgent._postproc(self, response)
        return result

    def process_stream_response(self, response):
        if response is None:
            return
        if isinstance(response, str):
            if not self.cancel_processing_var:
                yield response
        else:
            stream = get_sentence_from_stream_response(response)
            can_close = hasattr(response, "close")
            for i, sentence in enumerate(stream):
                if (i == 0) and (self.cancel_processing_var or self.interrupt_processing_var):
                    self.log.debug("Stream processing has been canceled")
                    if can_close:
                        response.close()
                    break
                yield sentence

    def set_canceling_variable(self, value):
        with self.lock:
            self.cancel_processing_var = value

    def set_interrupt_variable(self, value):
        with self.lock:
            self.interrupt_processing_var = value

    def set_processing_val(self, val):
        with self.lock:
            self.processing = val

    def is_empty_message(self, message):
        if isinstance(message, ExpressionAtom):
            children = message.get_children()
            if len(children) > 1 and str(get_grounded_atom_value(children[-1])).strip() == "":
                return True
        elif isinstance(message, str) and message.strip() == "":
            return True
        if message is None:
            return True

        return False

    def start(self, *args):
        super().start(*args)
        st = StreamMethod(self.messages_processor, args)
        st.start()

    def messages_processor(self, *args):
        # `*args` received on `start`
        while self.running:
            # TODO? func can be a Python function?
            message = self.messages.get()
            if message.message == self.stop_message:
                break
            self.said = False
            self.set_canceling_variable(False)
            self.set_interrupt_variable(False)
            for r in self.message_processor(message):
                self.outputs.put(r)

    def message_processor(self, input: AgentArgs):
        '''
        Takes a single message and provides a response if no canceling event has occurred.
        '''
        if self.is_empty_message(input.message):
            message = None
        elif str(input.message).startswith("(") and str(input.message).endswith(")"):
            message = input.message
        else:
            message = f"(user \"{input.message}\")"
        if (message is None) and (input.additional_info is None):
            self.set_processing_val(False)
            return

        self.set_processing_val(True)
        response = super().__call__(message, input.functions, input.additional_info).content
        for resp in self.process_stream_response(response):
            # cancel processing of the current message and return the message to the input
            if self.cancel_processing_var:
                self.log.info(f"message_processor:cancel processing for message {message}\n")
                self.input(input.message)
                break

            if self.interrupt_processing_var:
                self.log.info(f"message_processor:interrupt processing for message {message}\n")
                resp = "..."
            history_dict = {"event": "sentence",
                            "role": assistant_role,
                            "is_stream": False,
                            "user_id": None,
                            "message": resp}

            self.add_history(event_time=datetime.now(), history_dict=history_dict)
            #self._get_history()
            self.log.info(f"message_processor: return response for message {message} : {resp}")
            yield resp if input.language is None else (resp, input.language)
            # interrupt processing
            if self.interrupt_processing_var:
                break
        self.set_processing_val(False)

    def get_args(self, arg):
        args = []
        if hasattr(arg, "get_children"):
            for ch in arg.get_children():
                args.append(get_grounded_atom_value(ch))
        else:
            args.append(get_grounded_atom_value(arg))
        return args

    def input(self, args):

        msg = args[0]
        if 'data' in msg:
            msg = msg['data']
        if isinstance(msg, str):
            msg = {"message": msg}
        if len(args) > 1:
            msg['user_id'] = args[1]

        self.messages.put(AgentArgs(**msg))

    def create_event_message(self, event_name,  user_id=None):
        if event_name in system_event_description:
            message = system_event_description[event_name]
        else:
            message = f"[Event] {event_name}"
        if user_id is not None:
            message = message + f" [User_Id]: {user_id}"
        return message

    def handle_speechstart(self, arg):
        args = self.get_args(arg)
        self.speech_start = args[0]
        history_dict = {"event": "speechstart", "role": "system", "user_id": args[1] if len(args) > 1 else None}
        self.add_history(event_time=self.speech_start,
                         history_dict=history_dict)
        if self.processing:
            self.set_canceling_variable(not self.said)
        return []

    def handle_speechcont(self, arg):
        args = self.get_args(arg)
        tm = args[0]
        history_dict = {"event": "speechcont", "role": "system", "user_id": args[1] if len(args) > 1 else None}
        self.add_history(event_time=tm, history_dict=history_dict)
        if self.processing and self.said and ((time.time() - self.speech_start.timestamp()) > 0.5):
            self.set_interrupt_variable(True)
        return []

    def handle_speech(self, data):
        args = self.get_args(data)
        self.input(args)
        history_dict = {"event": "speech", "role": "system", "user_id": args[1] if len(args) > 1 else None}
        self.add_history(event_time=datetime.now(), history_dict=history_dict)

        return []

    def stop(self):
        super().stop()
        self.messages.put(AgentArgs(message=self.stop_message))

    def say(self):
        respond = []
        while not self.outputs.empty():
            with self.lock:
                self.said = True
                respond.append(ValueAtom(self.outputs.get()))
        return respond

    def has_output(self):
        return not self.outputs.empty()


@register_atoms(pass_metta=True)
def listening_gate_atoms(metta):
    return {
        r"listening-agent": OperationAtom('listening-agent',
                                          lambda path=None, event_bus=None:
                                          ListeningAgent.get_agent_atom(None,
                                                                        unwrap=False,
                                                                        path=path,
                                                                        event_bus=event_bus),
                                          unwrap=False),
    }
