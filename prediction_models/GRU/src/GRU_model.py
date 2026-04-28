import torch
import torch.nn as nn
from GRU_cell import Gru_Cell as GRUCell
"""
Ciekawostki dla bezrobotnych, działanie modelu GRU:
- przechodzimy sobie po kolejnych krokach czasowych sekwencji
- dla każdego kroku czasowego przechodzimy po kolejnych warstwach GRU
-każda wartwa aktualizuje swój stan ukryty na podstawie swojego poprzedniego stanu ukrytego
-wynik ostatniej warsty trafia od warstwy liniowej, która daje nam output dla danego kroku czasowego

Parametry:
    input_size : Liczba cech wejściowych w pojedynczym wektorze x_t
    hidden_size : Rozmiar stanu ukrytego h_t w każdej warstwie GRU
    output_size :Liczba cech na wyjściu modelu dla każdego kroku czasowego
    num_layers : Liczba warstw GRU ułożonych jedna na drugiej

kształt tensorów (jak coś tensor to po prostu wielowymiarowa tablica):
 Wejście:
    - x  : (batch_size, seq_len, input_size)
    - h0 : (num_layers, batch_size, hidden_size)

    Wyjście:
    - outputs : (batch_size, seq_len, output_size)
    - hn      : (num_layers, batch_size, hidden_size)

"""

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ModuleList to lista z pytorcha która ma podmoduły modelu
        # dzieki temu parametry wszystkich komorek GRU sa widoczne dla optymalizatora
        # no musi tak być, bo inaczej optymalizator by nie wiedział że ma aktualizować te parametry podczas treningu
        #gdybyśmy użyli zwykłej listy, Pytorch nie traktowałby tych warstw jako część modelu, więc ich parametry nie byłyby aktualizowane podczas treningu
        self.layers = nn.ModuleList()

        # ta pentelka for tworzy kojne warstwy GRU.
        #
        # Pierwsza warstwa dostaje bezpośrednio dane wejściowe x_t,
        # więc jej wejście ma rozmiar input_size.
        #
        # Każda następna warstwa dostaje jako wejście stan ukryty
        # z poprzedniej warstwy, więc jej wejście ma rozmiar hidden_size.
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(GRUCell(in_size, hidden_size))

        # warstwa liniowa na wyjściu czyli końccowa
        # po przejsciu przez wszystkie bierzemy wynik ostatniej i
        # dajemy do warstwy liniowej która daje nam output dla danego kroku czasowego
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """
        bede sie odnosiła do tych numerków i rozwijała
        co robimy w forwardzie:
        1. sprawdzamy rozmiary wejścia
        2. inicjalizujemy stany ukryte (jeśli h0 jest None, to tworzymy tensor zerowy)
        3. przechodzimy po kolejnych krokach czasowych sekwencji
        4. dla każdego kroku czasowego przechodzimy po kolejnych warstwach GRU
        5. każda warstwa aktualizuje swój stan ukryty na podstawie swojego poprzedniego stanu ukrytego
        6. wynik ostatniej warsty trafia od warstwy liniowej, która daje nam output dla danego kroku czasowego
        7. składamy wyniki z wszystkich kroków czasowych w jeden tensor
        - zwracamy tensor z wynikami dla wszystkich kroków czasowych oraz końcowe stany ukryte
        """

        #1. sprawdzamy rozmiary wejścia x to tensor o kształcie (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        #2. inicjalizujemy stany ukryte (jeśli h0 jest None, to tworzymy tensor zerowy)
        # h to lista, gdzie h[halyer] to aktualny stan ukryty konkrtnej warstwy
        #Każdy taki stan ukryty ma kształt (batch_size, hidden_size)
        if h0 is None:
            h = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
             # h0 przychodzi jako jeden tensor o kształcie:
            # (num_layers, batch_size, hidden_size)
            # Zamieniamy go na listę, żeby łatwo aktualizować stan
            # każdej warstwy osobno w pętli.
            h = [h0[i] for i in range(self.num_layers)]

        #lista do przechowywania wyników z każdego kroku czasowego
        outputs = []
        #3. przechodzimy po kolejnych krokach czasowych sekwencji
        for t in range(seq_len):
            # # Pobieramy dane wejściowe dla chwili t
            # dla całego batcha jednocześnie.
            #
            # x[:, t, :] ma kształt:
            # (batch_size, input_size)
            layer_input = x[:, t, :]
    #4. dla każdego kroku czasowego przechodzimy po kolejnych warstwach GRU
            for layer in range(self.num_layers):
                #ta linia kodu pod jest równoważna z tym h_t = GRUCell(x_t, h_prev)
                #bo self.layers[layer] to lista obiektór GRUcell i tutaj mamy niejawne wywołanie metody forward tej komórki, która aktualizuje stan ukryty tej warstwy.
                # czyli coś takiego self.layers[layer].forward(layer_input, h[layer])
                # gdzie:
                # - x_t to wejście do aktualnej warstwy w chwili t,
                # - h_prev to poprzedni stan ukryty tej warstwy.
                #5. każda warstwa aktualizuje swój stan ukryty na podstawie swojego poprzedniego stanu ukrytego
                h[layer] = self.layers[layer](layer_input, h[layer])
                # kolejna warstwa nie dostaje już surowego wejścia x_t,
                # tylko reprezentację przetworzoną przez warstwę niższą.
                layer_input = h[layer]

            # 6.warstwa liniowa
            # Po przejściu przez wszystkie warstwy GRU
            # zamieniamy wynik ostatniej warstwy na wyjście modelu.
            y = self.fc(layer_input)
            # Zapisujemy wyjście z bieżącego kroku czasowego.
            outputs.append(y)

        # 7.składanie wyników
        # Łączymy listę wyników w jeden tensor.
        # Przed stack:
        # outputs to lista długości seq_len,
        # a każdy element ma kształt (batch_size, output_size)
        # Po stack(dim=1):
        # otrzymujemy tensor o kształcie:
        # (batch_size, seq_len, output_size)
        outputs = torch.stack(outputs, dim=1)

        # końcowe stany ukryte
        # Łączymy końcowe stany ukryte wszystkich warstw.
        # Przed stack:
        # h to lista długości num_layers,
        # a każdy element ma kształt (batch_size, hidden_size)
        # Po stack(dim=0):
        # otrzymujemy tensor o kształcie:
        # (num_layers, batch_size, hidden_size)
        hn = torch.stack(h, dim=0)

        return outputs, hn
