import copy
import pylab
import random
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self):
        self.load_model = False         # True == 저장된 모델로 플레이, False == 학습
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]     # 상, 하, 좌, 우, 정지
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15

        self.discount_factor = 0.99     # 감가율
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01

        self.model = self.build_model()     # 모델 변수, 모델을 이용해 큐함수의 값을 얻음

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/deep_sarsa.h5')

    # 상태가 입력 큐함수가 출력인 인공신경망 생성
    # Dense 레이어 : 입출력을 모두 연결해줌. 전결합층
    # Dense 클래스
    # Dense(출력 뉴런의 수, input_dim=입력 뉴런의 수, init='가중치초기화방법', activation='활성화함수설정')
    # 가중치 초기화 방법
    # 'uniform' : 균일 분포, 'normal' : 가우시안 분포
    # 활성화 함수 설정
    # 'linear' : 디폴트값, 입력뉴런과 가중치로 계산된 결과값 그대로 출력
    # 'relu','sigmoid','softmax'
    def build_model(self):
        # 모델 구성
        # input 뉴런 수 = 15 (상태특징벡터의 수)
        # hidden Layer1 뉴런 수 = 30, hidden Layer2 뉴런 수 = 30
        # output 뉴런의 수 = 5 (행동의 수)
        model = Sequential()            # Sequential 모듈 : add 함수를 사용해 layer를 붙일 수 있음
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.summary()     # 모델의 요약 표현을 출력. utils.print_summary 의 shortcut ??

        # 모델 학습과정 설정
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 입실론 탐욕 방법으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return random.randrange(self.action_size)
        else:
            # 모델로부터 행동 산출
            state = np.float32(state)       # 케라스 모델에 들어가는 입력은 float 형태
            q_values = self.model.predict(state)        # model.predict(x) 는 입력 x 를 집어넣어 출력을 반환하는 함수
            return np.argmax(q_values[0])       # model.predict(state)의 출력형태 = [1,5]

    # 샘플을 가지고 인공신경망을 업데이트
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        # 살사의 큐함수 업데이트 식
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

        # 출력 값 reshape
        target = np.reshape(target, [1, 5])
        # 인공신경망 업데이트
        self.model.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    agent = DeepSARSAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])      # state를 1행 15열로 변환

        while not done:
            # env 초기화
            global_step += 1

            # 1. 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 2. 선택한 행동으로 환경에서 한 타임스텝 진행
            # 3. 환경으로부터 샘플(다음 상태, 보상, 성공여부) 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])
            # 4. 다음 상태에 대한 행동 선택
            next_action = agent.get_action(next_state)
            # 5. 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, next_action, done)

            state = next_state      # 얕은 복사 (메모리 주소 공유)
            score += reward

            state = copy.deepcopy(next_state)       # 깊은 복사 서로 다른 리스트

            if done:
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep_sarsa_.png")
                print("episode:", e, "  score:", score, "global_step", global_step, "  epsilon:", agent.epsilon)

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.model.save_weights("./save_model/deep_sarsa.h5")
