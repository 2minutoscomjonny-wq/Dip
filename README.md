import pandas as pd
import numpy as np
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
import requests
import io
import sys

class FootballPredictor:
    def __init__(self):
        self.fifa_url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        self.cbf_url = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master/campeonato-brasileiro-full.csv"
        self.df = None
        self.model = None

    def load_data(self):
        print("Carregando dados históricos (FIFA e CBF)...")
        try:
            # Dados FIFA
            r_fifa = requests.get(self.fifa_url, timeout=10)
            df_fifa = pd.read_csv(io.StringIO(r_fifa.text))
            df_fifa['date'] = pd.to_datetime(df_fifa['date'])
            # Filtrar jogos desde 2018 para refletir a fase atual das seleções
            df_fifa = df_fifa[df_fifa['date'].dt.year >= 2018]
            df_fifa = df_fifa[['home_team', 'away_team', 'home_score', 'away_score']]
            df_fifa.columns = ['home', 'away', 'home_score', 'away_score']

            # Dados CBF (Brasileirão)
            r_cbf = requests.get(self.cbf_url, timeout=10)
            try:
                df_cbf = pd.read_csv(io.StringIO(r_cbf.text))
            except:
                df_cbf = pd.read_csv(io.StringIO(r_cbf.content.decode('latin1')))
            
            df_cbf = df_cbf[['mandante', 'visitante', 'mandante_Placar', 'visitante_Placar']]
            df_cbf.columns = ['home', 'away', 'home_score', 'away_score']
            df_cbf = df_cbf.dropna()

            self.df = pd.concat([df_fifa, df_cbf], ignore_index=True)
            print(f"Sucesso! {len(self.df)} partidas carregadas.")
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            sys.exit(1)

    def train_model(self):
        print("Treinando modelo de Regressão de Poisson (isso pode levar alguns segundos)...")
        # Filtrar times com volume mínimo de jogos para evitar distorções estatísticas
        all_teams = pd.concat([self.df['home'], self.df['away']])
        team_counts = all_teams.value_counts()
        relevant_teams = team_counts[team_counts >= 10].index
        
        filtered_df = self.df[self.df['home'].isin(relevant_teams) & self.df['away'].isin(relevant_teams)]
        
        # Preparar formato para o modelo
        goal_model_data = pd.concat([
            filtered_df[['home', 'away', 'home_score']].rename(
                columns={'home': 'team', 'away': 'opponent', 'home_score': 'goals'}
            ).assign(home=1),
            filtered_df[['away', 'home', 'away_score']].rename(
                columns={'away': 'team', 'home': 'opponent', 'away_score': 'goals'}
            ).assign(home=0)
        ])

        self.model = smf.glm(formula="goals ~ home + team + opponent", 
                              data=goal_model_data, 
                              family=sm.families.Poisson()).fit()
        print("Modelo pronto para previsões!")

    def predict_exact_score(self, home_team, away_team, max_goals=5):
        try:
            # Prever médias de gols
            home_avg = self.model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team, 'home': 1}, index=[1])).values[0]
            away_avg = self.model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team, 'home': 0}, index=[1])).values[0]

            # Calcular probabilidades para cada placar (0x0 até 5x5)
            prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    prob_matrix[i, j] = poisson.pmf(i, home_avg) * poisson.pmf(j, away_avg)

            # Obter os resultados mais prováveis
            results = []
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    results.append(((i, j), prob_matrix[i, j]))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:5], home_avg, away_avg
        except Exception:
            return None, None, None

def main():
    predictor = FootballPredictor()
    predictor.load_data()
    predictor.train_model()

    print("\n" + "="*40)
    print("SISTEMA DE PREVISÃO DE PLACAR EXATO")
    print("Fontes: FIFA (Internacional) & CBF (Brasileirão)")
    print("="*40)

    while True:
        print("\nDigite os times ou 'sair':")
        home = input("Time da Casa: ").strip()
        if home.lower() == 'sair': break
        away = input("Time Visitante: ").strip()
        if away.lower() == 'sair': break

        top_5, h_avg, a_avg = predictor.predict_exact_score(home, away)

        if top_5:
            print(f"\nExpectativa de Gols: {home} ({h_avg:.2f}) vs {away} ({a_avg:.2f})")
            print("-" * 30)
            print("TOP 5 PLACARES MAIS PROVÁVEIS:")
            for (h_score, a_score), prob in top_5:
                print(f"Placar: {h_score} x {a_score}  --> Probabilidade: {prob*100:.2f}%")
        else:
            print(f"\nErro: Um dos times ('{home}' ou '{away}') não foi encontrado ou não tem dados suficientes.")
            print("Dica: Use nomes em inglês para seleções (ex: Brazil, Argentina) e nomes padrão para clubes (ex: Flamengo, Palmeiras).")

if __name__ == "__main__":
    main()
