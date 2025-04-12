# Raport wyników lista 5

## Nikodem Świerkowski

Poniższa tabela przedstawia wyniki procesu DVC:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Train Accuracy</th>
      <th>Train F1 Score</th>
      <th>Test Accuracy</th>
      <th>Test F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dummy</td>
      <td>0.622</td>
      <td>0.4770</td>
      <td>0.628</td>
      <td>0.4845</td>
    </tr>
    <tr>
      <td>SVM (Best Config)</td>
      <td>0.877</td>
      <td>0.8729</td>
      <td>0.554</td>
      <td>0.5304</td>
    </tr>
    <tr>
      <td>Random Forest (Best Config)</td>
      <td>0.900</td>
      <td>0.8982</td>
      <td>0.512</td>
      <td>0.5050</td>
    </tr>
  </tbody>
</table>

Spośród wszystkich modeli najlepiej zaprezentował się model Random Forest, który na zbiorze testowym uzyskał wynik 53% w F1, ale należy zwrócić uwagę że model Dummy uzyskał wynik niewiele gorszy 48% i uzyskał zdecydowanie najlepszy wynik w dokładności 63%. W konsekwencji możemy zaobserwować że wprowadzone przetworzenie danych nie pomogło w wystarczający sposób w uzyskaniu lepszych wyników, nawet jeśli f1 wypada lepiej w poważnych modelach.

Jeśli chodzi o konfiguracje modeli, wyglądają następująco:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Mode</th>
      <th>selectKBest</th>
      <th>PCA</th>
      <th>F1 Score (CV)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SVM</td>
      <td>all</td>
      <td>0</td>
      <td>1</td>
      <td>0.2761</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>non-text</td>
      <td>1</td>
      <td>0</td>
      <td>0.2367</td>
    </tr>
  </tbody>
</table>

F1 odnosi się do uzyskanej średniej wartości w procesie K-Fold. Oba modele inaczej reagowały na dane. SVM wypadł lepiej kiedy działał na całym zbiorze, ale po redukcji wymiarów z użyciem PCA. 
Random Forest, natomiast działał najlepiej po eliminacji kolumn tekstowych, a także wybraniu algorytmem selectKBest 10 najistotniejszych kolumn. 

Wiedza domenowa okazała się przydatna przez dodanie dodatkowych kolumn tekstowych w skrypcie raw_data_processing.py - kolumna zliczająca wykrzykniki była najbardziej skorelowana z docelową kolumną, pokazuje to że choć analiza tekstowa nie przyniosła oczekiwanych wartości w czasie treningu modeli to posiada ona duży potencjał w bardziej zaawansowanej analizie, która okazała się niemożliwa w tym projekcie z uwagi na ograniczenia technologiczne.  