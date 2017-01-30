# Przetwarzanie Cyfrowe Obrazów (POBR) - Projekt - Rozpoznawanie loga Tesco
Projekt wykonany w ramach przedmiotu POBR (Przetwarzanie Cyfrowe Obrazów) w semestrze 2016Z (studia magisterskie), na kierunku Informatyka, specjalizacji Inżynieria Systemów Informacyjnych (ISI) na Wydziale Elektroniki i Technik Informacyjnych (EiTI) Politechniki Warszawskiej.

**Informacje o projekcie**

* **Prowadzący projekt**: dr inż. Tomasz Trzciński
* **Ocena**: 4/10 (skala: -10/10)

Opis projektu
---
Dla indywidualnie wybranej klasy obrazów dobrać, zaimplementować i przetestować odpowiednie procedury wstępnego przetworzenia, segmentacji, wyznaczania cech oraz identyfikacji obrazów cyfrowych. Powstały w wyniku projektu program powinien poprawnie rozpoznawać wybrane obiekty dla reprezentatywnego zestawu obrazów wejściowych.

Wybrana klasa obrazów to zdjęcia zawierające logo sieci hipermarketów **Tesco** - czerwony tekst składający się z kolejno ułożonych liter T, E, S, C, O. Zestaw obrazów (``data/``) składa się z pięciu plików, z czego na każdym z nich znajdują się przynajmniej dwa fragmenty do rozpoznania.

Kompilacja i uruchomienie
---

### Wymagania
* Kompilator wspierający:
  * [Nested namespace definition](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4230.html)
    * GCC: 6.0
    * Clang: 3.6
    * MSVC: 14.3
* Narzędzie **Scons** (``scons``) lub **CMake** (``cmake``)
* Biblioteka **OpenCV 3.2.0**
  * [Automatyczna instalacja dla Ubuntu](https://github.com/jayrambhia/Install-OpenCV)

### Instrukcja
* **Scons**
  * Kompilacja: ``scons``
  * Uruchomienie: ``./build/run``
* _Dostępna również kompilacja w środowisku CLion_

### Testowane na:
* ``Ubuntu 16.04LTS`` + ``Clang 3.8.0-2ubuntu4``
