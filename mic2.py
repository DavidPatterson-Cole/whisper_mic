import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import queue
import tempfile
import os
import threading
import click
import torch
import numpy as np
import openai
from dotenv import load_dotenv, find_dotenv


# @click.command()
# @click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
# @click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
# @click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
# @click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
# @click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic engergy", type=bool)
# @click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
# @click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)

class Whisper_Microphone:
  def __init__(self, model="base", english=True,verbose=False, energy=300, pause=0.8,dynamic_energy=False,save_file=False):
      self.model = model
      self.english = english
      self.verbose = verbose
      self.energy = energy
      self.pause = pause
      self.dynamic_energy = dynamic_energy
      self.save_file = save_file
      self.audio_queue = queue.Queue()
      self.result_queue = queue.Queue()
      self.mic_start()

  def mic_start(self):
      temp_dir = tempfile.mkdtemp() if self.save_file else None
      #there are no english models for large
      if self.model != "large" and self.english:
          model = self.model + ".en"
      self.audio_model = whisper.load_model(self.model)
      # self.record_audio(self)
      # self.transcribe_forever(self, audio_model)
      threading.Thread(target=self.record_audio).start()
      threading.Thread(target=self.transcribe_forever).start()
      threading.Thread(target=self.gpt_forever).start()
      # print('result queue: ', result_queue.get())
      # return self.result_queue.get()
      # while True:
      #     print(self.result_queue.get())

  prompt = """
  
  Your job is to answer questions in a similar style to ChatGPT. 

  Examples of questions you might be asked:

  EXAMPLE QUESTION: Explain quantum computing in simple terms?
  EXAMPLE ANSWER: Quantum computing is a type of computing where information is processed using quantum-mechanical phenomena, such as superposition and entanglement. This allows quantum computers to perform certain calculations much faster than traditional computers. Instead of using bits, like traditional computers, quantum computers use quantum bits or qubits. These qubits can exist in multiple states at the same time, which allows quantum computers to perform multiple calculations at the same time. This makes them particularly useful for certain types of complex calculations, such as breaking encryption codes and simulating quantum systems.

  EXAMPLE QUESTION: Got any creative ideas for a 10 year old's birthday?
  EXAMPLE ANSWER: Here are a few creative ideas for a 10 year old's birthday: 1) Host a scavenger hunt around the neighborhood or in the park. 2) Have a themed party, such as a movie or book character. 3) Set up a DIY photo booth with fun props and costumes. 4) Have a baking or cooking party where the guests make their own pizzas or cupcakes. 5) Plan a day trip to a nearby attraction, such as an amusement park or zoo. 6) Have a game night with board games and group games. 7) Host a karaoke party with a karaoke machine or virtual karaoke app. 8) Have a craft party where the guests can make their own jewelry, tie-dye shirts, or friendship bracelets.

  EXAMPLE QUESTION: How do I make an HTTP request in Javascript?
  EXAMPLE ANSWER: There are several ways to make an HTTP request in JavaScript, some of the most popular methods include: 1) Using the XMLHttpRequest object: This is the most common method for making HTTP requests in JavaScript. It allows you to send and receive data from a server asynchronously. 2) Using the fetch() method: This is a more recent method for making HTTP requests in JavaScript. It uses Promises and is considered more modern and easier to use compared to the XMLHttpRequest object.

  The question you need to answer is below:
  ---
  QUESTION: $question ?
  ANSWER:

  """

  def gpt_forever(self):
    load_dotenv(find_dotenv())
    openai.api_key = os.getenv('openai_api_key')
    queue_size = self.result_queue.qsize()
    while True:
      new_queue_size = self.result_queue.qsize()
      if new_queue_size > queue_size:
        queue_size = new_queue_size
        print("You said: ", self.result_queue.get())
        prompt = self.prompt.replace("$question", self.result_queue.get())
        print("Prompt: ", self.prompt)
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, top_p=1, temperature=0.3, best_of=1, n=1, max_tokens=250)
        print("Response: ", response.choices[0].text)


  def record_audio(self):
      #load the speech recognizer and set the initial energy threshold and pause threshold
      r = sr.Recognizer()
      r.energy_threshold = self.energy
      r.pause_threshold = self.pause
      r.dynamic_energy_threshold = self.dynamic_energy

      with sr.Microphone(sample_rate=16000) as source:
          print("Say something!")
          i = 0
          while True:
              #get and save audio to wav file
              audio = r.listen(source)
              # if self.save_file:
              #     data = io.BytesIO(audio.get_wav_data())
              #     audio_clip = AudioSegment.from_file(data)
              #     filename = os.path.join(temp_dir, f"temp{i}.wav")
              #     audio_clip.export(filename, format="wav")
              #     audio_data = filename
              # else:
              torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
              audio_data = torch_audio
              # print('len audio_queue 1: ', len(audio_queue))
              self.audio_queue.put_nowait(audio_data)
              # audio_queue.append(audio_data)
              # print('len audio_queue 2: ', len(audio_queue))
              i += 1


  def transcribe_forever(self):
      while True:
          audio_data = self.audio_queue.get()
          # audio_data = audio_queue[0] if len(audio_queue) > 0 else None
          if self.english:
              result = self.audio_model.transcribe(audio_data,language='english')
          else:
              result = self.audio_model.transcribe(self.audio_data)

          if not self.verbose:
              predicted_text = result["text"]
              self.result_queue.put_nowait(predicted_text)
          else:
              self.result_queue.put_nowait(result)

          if self.save_file:
              os.remove(audio_data)


Whisper_Microphone()
