import os
import time
from pydub import AudioSegment
import speech_recognition as sr
from textblob import TextBlob
from multiprocessing import Pool
import matplotlib.pyplot as plt
from deepmultilingualpunctuation import PunctuationModel

class AudioProcessor:
    """
    Clase para manejar la carga, conversión y división de audio en bloques procesables.
    """
    def __init__(self, audio_path, silence_thresh=-50, block_duration_ms=10000):
        self.audio_path = audio_path
        self.silence_thresh = silence_thresh  # Umbral de decibelios en dBFS
        self.block_duration_ms = block_duration_ms  # Duración de los bloques en milisegundos (10 segundos)
        self.audio = None
        self.converted_path = "audio_temp.wav"
        self.duration = 0

    def load_audio(self):
        """Carga un archivo de audio y obtiene su duración."""
        print("Cargando archivo de audio...")
        ext = os.path.splitext(self.audio_path)[1].lower()
        self.audio = AudioSegment.from_file(self.audio_path, format=ext.lstrip('.'))
        self.duration = len(self.audio)

    def convert_audio(self):
        """Convierte el audio cargado a formato WAV."""
        print("Convirtiendo archivo de audio a WAV...")
        self.audio.export(self.converted_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])

    def get_audio_blocks(self):
        """Divide el audio en bloques de duración fija (10 segundos) pero con cortes según los decibeles."""
        print("Dividiendo el audio en bloques de 10 segundos...")
        blocks = []
        start_ms = 0

        while start_ms < self.duration:
            end_ms = start_ms + self.block_duration_ms
            block = self.audio[start_ms:end_ms]
            
            # Verificar si el bloque final tiene decibeles por encima de -50 dB
            while block.dBFS > self.silence_thresh and end_ms < self.duration:
                # Esperar hasta que los decibeles caigan por debajo del umbral
                end_ms += 500  # Aumentar en intervalos de 500 ms
                block = self.audio[start_ms:end_ms]
            
            blocks.append(block)
            start_ms = end_ms  # Mover al siguiente bloque

        return blocks

class SpeechRecognizer:
    """
    Clase para procesar audio y generar texto con puntuación utilizando SpeechRecognition.
    """
    def __init__(self):
        self.texto_completo = ""
        self.recognizer = sr.Recognizer()
        self.punctuation_model = PunctuationModel()

    def process_block(self, audio_segment, block_index):
        """Procesa un bloque de audio y extrae texto utilizando SpeechRecognition."""
        try:
            print(f"Transcribiendo bloque {block_index}...")
            # Crear un buffer de audio a partir de los datos crudos (raw data) del segmento de audio
            audio_data = sr.AudioData(audio_segment.raw_data, audio_segment.frame_rate, audio_segment.sample_width)
            # Realizar la transcripción usando el reconocedor de Google
            text = self.recognizer.recognize_google(audio_data, language="es-ES")
        except sr.UnknownValueError:
            print(f"No se pudo entender el audio en el bloque {block_index}.")
            text = ""
        except sr.RequestError as e:
            print(f"Error de solicitud a Google Speech API: {e}")
            text = ""
        
        return text

    def process_audio_blocks(self, audio_blocks):
        """Procesa todos los bloques de audio y construye el texto completo."""
        if not audio_blocks:
            print("Error: No se detectaron bloques de audio para procesar.")
            return
        
        start_time = time.time()
        # Limitar el número de procesos en paralelo
        num_processes = min(4, len(audio_blocks))  # Limitar el número de procesos al número de bloques o 4
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(self.process_block, [(block, idx) for idx, block in enumerate(audio_blocks)])
        
        # Unir los resultados y medir el tiempo de procesamiento
        self.texto_completo = " ".join(results)
        end_time = time.time()
        print(f"Tiempo total de extracción de texto: {end_time - start_time:.3f} segundos")

    def punctuate_complete_text(self):
        """Agrega puntuación al texto completo."""
        print("Agregando puntuación al texto completo...")
        self.texto_completo = self.punctuation_model.restore_punctuation(self.texto_completo)
        print("Texto con puntuación añadido:")
        print(self.texto_completo)

    def analyze_complete_text(self):
        """Realiza un análisis de sentimiento sobre el texto completo utilizando TextBlob."""
        start_time = time.time()
        print("Analizando el texto completo...")
        analysis = TextBlob(self.texto_completo)
        sentiment_polarity = analysis.sentiment.polarity
        sentiment_label = "Positivo" if sentiment_polarity > 0 else "Negativo" if sentiment_polarity < 0 else "Neutral"
        end_time = time.time()
        print(f"Tiempo total de análisis de sentimiento: {end_time - start_time:.3f} segundos")
        print(f"Análisis de sentimiento global: {sentiment_label} (Polaridad: {sentiment_polarity:.2f})")

class AudioVisualizer:
    """
    Clase para generar visualizaciones del audio procesado.
    """
    def __init__(self, audio):
        self.audio = audio

    def generate_visualization(self):
        """Genera un gráfico del nivel de decibeles a lo largo del tiempo."""
        print("Generando gráfico...")
        step = len(self.audio) // 1000
        times = list(range(0, len(self.audio), step))
        levels = [self.audio[i:i + step].dBFS for i in times]

        plt.figure(figsize=(15, 5))
        plt.plot([t / 1000 for t in times], levels, label="Nivel de decibeles (dB)")
        plt.title("Análisis de audio: Nivel de decibeles y pausas detectadas")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Decibeles (dB)")
        
        # Mostrar la rejilla
        plt.grid(True)

        # Mostrar ticks cada 0.5 segundos
        ticks = [i * 0.5 for i in range(0, int(self.audio.duration_seconds * 2) + 1)]
        plt.xticks(ticks)

        plt.tight_layout()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    try:
        audio_path = "a.m4a"
        processor = AudioProcessor(audio_path)
        processor.load_audio()
        processor.convert_audio()
        audio_blocks = processor.get_audio_blocks()

        if audio_blocks:  # Verificar si se generaron bloques antes de continuar
            # Procesar los bloques de audio
            recognizer = SpeechRecognizer()
            recognizer.process_audio_blocks(audio_blocks)

            # Puntuación y análisis
            recognizer.punctuate_complete_text()
            recognizer.analyze_complete_text()

            print("Procesamiento completado.")

            visualizer = AudioVisualizer(processor.audio)
            visualizer.generate_visualization()
        else:
            print("No se pudieron procesar bloques de audio.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")
