CXX = c++

CXXFLAGS = -Wall -Wextra -Werror -g -MMD

OBJS_DIR = obj

SRCS_TRAIN = 	src/ft_linear_regression.cpp \
				src/train.cpp

SRCS = 	src/ft_linear_regression.cpp \
		src/compute.cpp

DEPS = $(OBJS:.o=.d)

DEPS_TRAIN = $(OBJS_TRAIN:.o=.d)

OBJS_TRAIN = $(SRCS_TRAIN:%.cpp=$(OBJS_DIR)/%.o)

OBJS = $(SRCS:%.cpp=$(OBJS_DIR)/%.o)

NAME = compute

NAME_TRAIN = train

$(NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(NAME_TRAIN): $(OBJS_TRAIN)
	$(CXX) $(CXXFLAGS) $^ -o $@
	
$(OBJS_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

compute: $(NAME)

train: $(NAME_TRAIN)

clean:
	rm -rf $(OBJS_DIR) $(DEPS) $(DEPS_TRAIN)

fclean: clean
	rm -f $(NAME) $(NAME_TRAIN)

-include $(DEPS)
-include $(DEPS_TRAIN)

.PHONY: compute train clean fclean