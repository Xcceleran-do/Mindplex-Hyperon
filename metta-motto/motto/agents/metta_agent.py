from datetime import datetime
import logging

from .agent import Agent, Response
from hyperon import *
from motto.utils import *
from motto.utils import get_ticks

class MettaAgent(Agent):
    def __metta_call__(self, *args):
        if len(args) > 0:
            if isinstance(args[0], SymbolAtom):
                # support fot calls like (&a1 .input "who is the 6 president of France")
                n = args[0].get_name()
                if n[0] == '.' and hasattr(self, n[1:]):
                    method = getattr(self, n[1:])
                    args = args[1:]
                    method = OperationObject(f"{method}", method).execute
                    return method(*args)
        return super().__metta_call__(*args)

    def _init_metta(self):
        super()._init_metta()
        # TODO: assert
        self._metta.run("!(import! &self motto)")
        self._metta.run("!(import! &self py_ops)")

    def _prepare(self, msgs_atom, additional_info=None):
        # The context space is recreated on each call
        if self._context_space is not None:
            self._metta.space().remove_atom(self._context_space)
        self._context_space = G(GroundingSpaceRef())
        self._metta.space().add_atom(self._context_space)
        context_space = self._context_space.get_object()
        if msgs_atom is not None:
            context_space.add_atom(E(S('='), E(S('messages')), msgs_atom))
        # what to do if need to set some variables from python?
        if additional_info is not None:
            for val in additional_info:
                f, v, t = val
                context_space.add_atom(
                    E(S(':'), S(f), E(S('->'), S(t))))
                context_space.add_atom(
                    E(S('='), E(S(f)), ValueAtom(v)))

    def _postproc(self, response):
        # No postprocessing is needed here
        return Response(response, None)

    def __call__(self, msgs_atom, functions=[], additional_info=None):
        # TODO: support {'role': , 'content': } dict input
        if isinstance(msgs_atom, str):
            msgs_atom = self._metta.parse_single(msgs_atom)
        self._prepare(msgs_atom, additional_info)
        response = self._metta.run('!(response)')
        return self._postproc(response[0])

    def get_state(self, state_name):
        response = self._metta.run(f'!(get-state {state_name})', flat=True)
        return repr(response[0])


class MettaScriptAgent(MettaAgent):

    def _create_metta(self):
        # Skipping _create_metta in super.__init__
        pass

    def __call__(self, msgs_atom, functions=[], additional_info=None):
        self._init_metta()
        if isinstance(msgs_atom, str):
            msgs_atom = self._metta.parse_single(msgs_atom)
        self._prepare(msgs_atom, additional_info)
        # Loading the code after _prepare
        super()._load_code()
        response = self._metta.run('!(response)')

        return self._postproc(response[0])


class DialogAgent(MettaAgent):
    def __init__(self, path=None, atoms={}, include_paths=None, code=None,event_bus=None):
        super().__init__(path, atoms, include_paths, code, event_bus)
        self.log = logging.getLogger(__name__ + '.' + type(self).__name__)

    def create_event_message(self, event_name, user_id=None):
        return None


    def add_history(self, event_time: datetime, history_dict: dict):
        message = None
        event = history_dict["event"] if "event" in history_dict else None
        user_id = history_dict["user_id"] if "user_id" in history_dict else None
        if ("message" not in history_dict) and (event is not None):
            message = self.create_event_message(event,  user_id=user_id)
            # todo should the system language be the same as user's language?
            language = "en"
        if "message" in history_dict:
            message = history_dict["message"]

        dict_space = GroundingSpaceRef()
        if message is not None:
            if isinstance(message, str) and ("role" in history_dict):
                dict_space.add_atom(E(S("message"), E(S(history_dict["role"]), G(ValueObject(message)))))
            else:
                dict_space.add_atom(E(S("message"), message))

        processed_keys = ["role", "message"]

        for k,v in history_dict.items():
            if k not in processed_keys:
                dict_space.add_atom(E(S(k), ValueAtom(v)))
        self._metta.space().add_atom(E(S("history"), ValueAtom(get_ticks(event_time)), G(dict_space)))

    def get_sentence_history(self):
        history_dicts = self._metta.run('''
                !(let ($time $sp)  (match &self (history $t $s) ($t $s))
                        (match  $sp (,(event "sentence") (message $message) (is_stream False) (user_id $user)) ($time $message $user))
                 )
                ''', True)

        t = [x.get_children() for x in history_dicts]
        return sorted(t, key=lambda x: x[0].get_object().value)

    def get_speech_history(self):
        history_dicts = self._metta.run('''
                ! (let ($time $sp)  (match &self (history $t $s) ($t $s))
                                    (match  $sp (,(event "speech") (message $message)(user_id $user)) ($time $message $user))
                  )
                ''', True)

        t = [x.get_children() for x in history_dicts]
        return sorted(t, key=lambda x: x[0].get_object().value)

    def get_user_id_info(self, history):
        user_id = None
        speech_hist = self.get_speech_history()
        if len(speech_hist) > 0:
            hist = speech_hist[-1]
            user_id = hist[2].get_object().value
            # add "speech" event to history only if we know the user_id
            if user_id is not None:
                history.append(hist[1])

        return user_id


    def _prepare(self, msgs_atom, additional_info=None):
        super()._prepare(msgs_atom, additional_info)

        history = []
        history_dicts = self.get_sentence_history()

        for hist in history_dicts:
            history.append(hist[1])
        user_id = self.get_user_id_info(history)

        self._context_space.get_object().add_atom(
            E(S('='), E(S('history')), E(S('Messages'), *history)))

        # atm, we put the input message into the history by default
        self.add_history(event_time=datetime.now(), history_dict={"event": "sentence",
                                                                  "is_stream": False,
                                                                  "user_id": user_id,
                                                                  "message": msgs_atom})

    def _postproc(self, response):
        # TODO it's very initial version of post-processing
        # The output is added to the history as the assistant utterance.
        # This might be ok: if someone wants to avoid this, Function
        # (TODO not supported yet) instead of Response could be used.
        # But one can add explicit commands for putting something into
        # the history as well as to do other stuff
        result = super()._postproc(response)
        # TODO: 0 or >1 results, to expression?
        resp_value = get_string(result.content[0])
        self.add_history(event_time=datetime.now(), history_dict={"event": "sentence",
                                                                  "role": assistant_role,
                                                                  "is_stream": not isinstance(resp_value, str),
                                                                  "user_id": None,
                                                                  "message":  resp_value})

        return result

    def clear_history(self):
        self._metta.run(
            f"!(let ($time $sp) (match &self (history $t $s) ($t $s)) \
            (remove-atom &self  (history $time $sp)))")

    def _get_history(self):
        history_dicts = self._metta.run(
            f"!(match &self (history $t $s) $s)")

        hist = history_dicts[0]
        for h in hist:
            try:
                print("hist:", h.get_object().get_atoms())
            except Exception as ex:
                print(ex)

    def get_response_by_index(self, index, role=assistant_role):
        history_dicts = self.get_sentence_history()
        if len(history_dicts) - 1 < index:
            return None
        hist = history_dicts[index][1]
        children = hist.get_children()
        if children[0].get_name() == role:
            return children[1].get_object().value
        return None

    def process_stream_response(self, response):
        if response is None:
            return
        if isinstance(response, str):
            yield response
        else:
            stream = get_sentence_from_stream_response(response)
            can_close = hasattr(response, "close")
            for i, sentence in enumerate(stream):
                self.add_history(event_time=datetime.now(), history_dict={"event": "sentence",
                                                                          "role": assistant_role,
                                                                          "is_stream": False,
                                                                          "user_id": None,
                                                                          "message": sentence})
                yield sentence

    def process_last_stream_response(self):
        response = self.get_response_by_index(-1)
        yield from self.process_stream_response(response)
