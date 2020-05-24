from matplotlib import pyplot as plt # matplot은

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]


plt.plot(years, gdp, color='green', marker='o', linestyle='solid') # 그래프 선 만들기 years x축과 gdp y축으로 설정한다 색과 마커화 선의 스타일도 설정 가능

plt.title("Nominal GDP") # 그래프에 제목을단다


plt.ylabel("Billions of $") # y축에 제목(라벨) 달아줌

plt.show() # 그래프 출력

