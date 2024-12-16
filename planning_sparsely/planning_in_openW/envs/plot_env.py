# @title Env plotting
from imports import *

import matplotlib.pylab as plt
import matplotlib.patches as patches

PLT_UP = np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5]])
PLT_LEFT = np.array([[0, 0], [-0.5, 0.5], [-0.5, -0.5]])
PLT_RIGHT = np.array([[0, 0], [0.5, 0.5], [0.5, -0.5]])
PLT_DOWN = np.array([[0, 0], [0.5, -0.5], [-0.5, -0.5]])

TXT_OFFSET_VAL = 0.3
TXT_CENTERING = np.array([-0.08, -0.05])
TXT_UP = np.array([TXT_OFFSET_VAL, 0]) + TXT_CENTERING
TXT_LEFT = np.array([0, -TXT_OFFSET_VAL]) + TXT_CENTERING
TXT_RIGHT = np.array([0, TXT_OFFSET_VAL]) + TXT_CENTERING
TXT_DOWN = np.array([-TXT_OFFSET_VAL, 0]) + TXT_CENTERING

OP_0 = np.array([0, 0.5])
OP_1 = np.array([0.5, 0.5])
OP_2 = np.array([0, 0])
OP_3 = np.array([0.5, 0])
OP_OFFSETS = [OP_0, OP_1, OP_2, OP_3]

ACT_OFFSETS = [
    [PLT_UP, TXT_UP],
    [PLT_RIGHT, TXT_RIGHT],
    [PLT_DOWN, TXT_DOWN],
    [PLT_LEFT, TXT_LEFT],
]

PLOT_CMAP = cm.Blues


def plot_sa_values(env, q_values, text_values=True,
                   invert_y=True,
                   update=False,
                   title=None):
    w = env._num_cols
    h = env._num_rows

    if update:
        clear_output(wait=True)
    plt.figure()
    ax = plt.gca()
    normalized_values = q_values
    normalized_values = normalized_values - np.min(normalized_values)
    normalized_values = normalized_values / np.max(normalized_values)
    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        if invert_y:
            y = h - y - 1

        xy = np.array([x, y])
        xy3 = np.expand_dims(xy, axis=0)

        for a in range(len(env.get_action_set())):
            val = normalized_values[state_idx, a]
            og_val = q_values[state_idx, a]

            patch_offset, txt_offset = ACT_OFFSETS[a]

            if text_values:
                xy_text = xy + txt_offset
                ax.text(xy_text[0], xy_text[1], '%.1f' % og_val, size='small')
            color = PLOT_CMAP(val)
            ax.add_patch(Polygon(xy3 + patch_offset, True, color=color))
    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))
    plt.grid()
    if title:
        plt.title(title)
    plt.show()


def plot_sa_values_3(env, q_values1, q_values2, q_values3, text_values=True,
                     invert_y=True,
                     update=False,
                     title=None,
                     fontsize=25,
                     subtitles=None):
    w = env._num_cols
    h = env._num_rows

    if update:
        clear_output(wait=True)
    fig = plt.figure(figsize=(21, 7))
    fig.suptitle(title, fontsize=fontsize)

    #   ax = plt.gca()
    for i, q_values in enumerate([q_values1, q_values2, q_values3]):
        ax = plt.subplot(1, 3, i + 1)
        normalized_values = q_values
        normalized_values = normalized_values - np.min(normalized_values)
        normalized_values = normalized_values / np.max(normalized_values)
        for x, y in itertools.product(range(w), range(h)):
            state_idx = env._get_state_idx(y, x)
            if invert_y:
                y = h - y - 1

            xy = np.array([x, y])
            xy3 = np.expand_dims(xy, axis=0)

            for a in range(len(env.get_action_set())):
                val = normalized_values[state_idx, a]
                og_val = q_values[state_idx, a]

                patch_offset, txt_offset = ACT_OFFSETS[a]

                if text_values:
                    xy_text = xy + txt_offset
                    ax.text(xy_text[0], xy_text[1], '%.1f' % og_val, size='small')
                color = PLOT_CMAP(val)
                ax.add_patch(Polygon(xy3 + patch_offset, True, color=color))
        ax.set_xticks(np.arange(-1, w + 1, 1))
        ax.set_yticks(np.arange(-1, h + 1, 1))
        ax.grid()
        ax.set_title(subtitles[i], fontsize=fontsize)
        fig.add_subplot(ax)
    plt.show()


def plot_sa_values_4(env, policy, option_interest,
                     q_values1, q_values2, text_values=True,
                     invert_y=True,
                     update=False,
                     title=None,
                     fontsize=25,
                     subtitles=None):
    w = env._num_cols
    h = env._num_rows

    if update:
        clear_output(wait=True)
    fig = plt.figure(figsize=(21, 7))
    fig.suptitle(title, fontsize=fontsize)

    for i, q_values in enumerate([q_values1, q_values2]):
        ax = plt.subplot(1, 3, i + 1)
        v_values = q_values[np.arange(q_values.shape[0]), np.argmax(policy, axis=-1)]
        normalized_values = q_values
        normalized_values = normalized_values - np.min(normalized_values)
        normalized_values = normalized_values / np.max(normalized_values)
        for x, y in itertools.product(range(w), range(h)):
            state_idx = env._get_state_idx(y, x)
            if invert_y:
                yy = h - y - 1

            xy = np.array([x, yy])
            xy3 = np.expand_dims(xy, axis=0)

            # for a in range(len(env.get_action_set())):
            val = normalized_values[state_idx]  # ,a]
            og_val = v_values[state_idx]  # ,a]

            # patch_offset, txt_offset = ACT_OFFSETS[a]

            # if text_values:
            #     xy_text = xy+txt_offset
            #     ax.text(xy_text[0], xy_text[1], '%.1f'%og_val, size='small')
            color = PLOT_CMAP(val)
            ax.add_patch(ax.add_patch(Rectangle(xy, 1, 1, color=color)))
            # Polygon(xy3+patch_offset+0.5, True, color=color))
            if env._matrix_mdp[y][x] == -1:
                ax.add_patch(
                    patches.Rectangle(
                        (x, yy),  # (x,y)
                        1.0,  # width
                        1.0,  # height
                        facecolor="gray"
                    )
                )

        # ax.set_xticks(np.arange(-1, w+1, 1))
        # ax.set_yticks(np.arange(-1, h+1, 1))
        ax.set_xlim([0, env._num_cols])

        ax.grid()
        ax.set_title(subtitles[i], fontsize=fontsize)

        ax.set_ylim([0, env._num_rows])

        for j in range(env._num_cols):
            plt.axvline(j, color='k', linestyle=':')
        plt.axvline(env._num_cols, color='k', linestyle=':')

        for j in range(env._num_rows):
            plt.axhline(j, color='k', linestyle=':')
        plt.axhline(env._num_rows, color='k', linestyle=':')
        fig.add_subplot(ax)

    v_values = np.max(q_values, -1)
    normalized_values = v_values
    normalized_values = normalized_values - np.min(normalized_values)
    normalized_values = normalized_values / np.max(normalized_values)
    ax = plt.subplot(1, 3, 3)
    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        a = np.argmax(policy[state_idx])

        if invert_y:
            yy = h - y - 1

        xy = np.array([x, yy])

        val = normalized_values[state_idx]
        og_val = v_values[state_idx]
        if text_values:
            xy_text = xy
            ax.text(xy_text[0], xy_text[1] - 1, '%.1f' % og_val, size='small')
        color = PLOT_CMAP(val)
        ax.add_patch(Rectangle(xy, 1, 1, color=color))

        dx = 0
        dy = 0
        if a == 0:  # up
            dy = 0.3
        elif a == 1:  # right
            dx = 0.3
        elif a == 2:  # down
            dy = -0.3
        elif a == 3:  # left
            dx = -0.3
        elif env._matrix_mdp[y][x] != -1 and option_interest[state_idx] == 0:  # termination
            circle = plt.Circle(
                (x + 0.5, env._num_rows - y + 0.5 - 1), 0.025, color='k')
            ax.add_artist(circle)
        if env._matrix_mdp[y][x] != -1 and option_interest[state_idx] == 1:
            plt.arrow(x + 0.5, env._num_rows - y + 0.5 - 1, dx, dy,
                      head_width=0.15, head_length=0.15, fc='k', ec='k')
        if env._matrix_mdp[y][x] == -1:
            ax.add_patch(
                patches.Rectangle(
                    (x, env._num_rows - y - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="gray"
                )
            )

    ax.set_xlim([0, env._num_cols])
    ax.set_ylim([0, env._num_rows])

    for i in range(env._num_cols):
        plt.axvline(i, color='k', linestyle=':')
    plt.axvline(env._num_cols, color='k', linestyle=':')

    for j in range(env._num_rows):
        plt.axhline(j, color='k', linestyle=':')
    plt.axhline(env._num_rows, color='k', linestyle=':')

    plt.grid()
    ax.set_title(subtitles[-1], fontsize=fontsize)
    fig.add_subplot(ax)
    plt.show()


def plot_sa_values_pio(env, q_pi, options_q_star, options_q_pi,
                       q_star, interest, option_interests,
                       text_values=True,
                       invert_y=True,
                       update=False,
                       title=None,
                       fontsize=25,
                       subtitles=None):
    w = env._num_cols
    h = env._num_rows

    if update:
        clear_output(wait=True)
    fig = plt.figure(figsize=(14, 14))
    fig.suptitle(title, fontsize=fontsize)

    ax = plt.subplot(1, 1, 1)

    for oo in range(len(option_interests)):
        q_values = options_q_pi[oo]
        normalized_values = q_values
        normalized_values = normalized_values - np.min(normalized_values)
        normalized_values = normalized_values / np.max(normalized_values)
        option_interest = option_interests[oo]
        for x, y in itertools.product(range(w), range(h)):
            state_idx = env._get_state_idx(y, x)

            if invert_y:
                yy = h - y - 1

            # xy = np.array([x, yy])
            xy = np.array([x, yy])
            xy3 = np.expand_dims(xy, axis=0)

            op_offset = OP_OFFSETS[oo]
            if interest[state_idx] > 0:
                # if text_values:
                xy_text = xy
                a = np.argmax(q_values[state_idx])

                dx = 0
                dy = 0
                if a == 0:  # up
                    dy = 0.1
                elif a == 1:  # right
                    dx = 0.1
                elif a == 2:  # down
                    dy = -0.1
                elif a == 3:  # left
                    dx = -0.1
                # elif env._matrix_mdp[y][x] != -1 and option_interest[state_idx] == 0: # termination
                #     circle = plt.Circle(
                #         (x + 0.5, env._num_rows - y + 0.5 - 1), 0.025, color='k')
                #     ax.add_artist(circle)
                plt.arrow(xy_text[0] + op_offset[0] + 0.1, xy_text[1] + op_offset[1] + 0.2, dx, dy,
                          head_width=0.1, head_length=0.1, fc=COLORS[oo % len(COLORS)], ec=COLORS[oo % len(COLORS)])

                ax.text(xy_text[0] + op_offset[0] + 0.3, xy_text[1] + op_offset[1] + 0.1, str(int(oo)), size='small',
                        c=COLORS[oo % len(COLORS)])
                color = PLOT_CMAP(val)
                ax.add_patch(Rectangle(xy + op_offset, 0.5, 0.5, color=color))

            if env._matrix_mdp[y][x] != -1 and interest[state_idx] == 0 and option_interest[state_idx] > 0:
                for aa in range(len(env.get_action_set())):
                    val = normalized_values[state_idx, aa]
                    patch_offset, txt_offset = ACT_OFFSETS[aa]
                    color = PLOT_CMAP(val)
                    ax.add_patch(Polygon(xy3 + patch_offset + 0.5, True, color=color))

                a = np.argmax(q_values[state_idx])

                dx = 0
                dy = 0
                if a == 0:  # up
                    dy = 0.3
                elif a == 1:  # right
                    dx = 0.3
                elif a == 2:  # down
                    dy = -0.3
                elif a == 3:  # left
                    dx = -0.3
                # elif env._matrix_mdp[y][x] != -1 and option_interest[state_idx] == 0: # termination
                #     circle = plt.Circle(
                #         (x + 0.5, env._num_rows - y + 0.5 - 1), 0.025, color='k')
                #     ax.add_artist(circle)
                plt.arrow(x + 0.5, env._num_rows - y + 0.5 - 1, dx, dy,
                          head_width=0.15, head_length=0.15, fc=COLORS[oo % len(COLORS)], ec=COLORS[oo % len(COLORS)])

    normalized_values = q_pi
    normalized_values = normalized_values - np.min(normalized_values)
    normalized_values = normalized_values / np.max(normalized_values)

    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        # o = np.argmax(policy[state_idx])
        o_i = []
        # for option, option_interest in enumerate(option_interests):
        #     o_i.append(option_interest[state_idx])
        # q = np.where(np.asarray(o_i) == 0, -np.inf, )
        q = q_pi[state_idx]
        o = np.argmax(q)
        if invert_y:
            yy = h - y - 1

        xy = np.array([x, yy])
        # xy4 = np.expand_dims(xy, axis=0)

        # val = normalized_values[state_idx][o]
        # xy_text = xy

        op_offset = OP_OFFSETS[o]
        # if interest[state_idx] > 0:
        #     # if text_values:
        #     #     xy_text = xy+txt_offset
        #     #     ax.text(xy_text[0], xy_text[1], '%.1f'%og_val, size='small')
        #     color = PLOT_CMAP(val)
        #     ax.add_patch(Rectangle(xy+op_offset, 0.5, 0.5, color=color))

        # Polygon(xy4+op_offset, True, color=color))
        # ax.text(xy_text[0] + 0.5, xy_text[1]+0.5, str(int(o)), size='small', color=COLORS[o % len(COLORS)])
        # color = PLOT_CMAP(val)
        # ax.add_patch(Rectangle(xy, 1, 1, color=color))
        if env._matrix_mdp[y][x] == -1:
            ax.add_patch(
                patches.Rectangle(
                    (x, env._num_rows - y - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="gray"
                )
            )

    ax.set_xlim([0, env._num_cols])
    ax.set_ylim([0, env._num_rows])

    for i in range(env._num_cols):
        plt.axvline(i, color='k', linestyle=':')
    plt.axvline(env._num_cols, color='k', linestyle=':')

    for j in range(env._num_rows):
        plt.axhline(j, color='k', linestyle=':')
    plt.axhline(env._num_rows, color='k', linestyle=':')

    plt.grid()
    ax.set_title(subtitles[-1], fontsize=fontsize)
    fig.add_subplot(ax)
    plt.show()


def plot_s_values(env, v_values, text_values=True,
                  invert_y=True, invert_x=False, update=False,
                  title=None):
    w = env._num_cols
    h = env._num_rows
    if update:
        clear_output(wait=True)
    plt.figure()
    ax = plt.gca()
    normalized_values = v_values
    normalized_values = normalized_values - np.min(normalized_values)
    normalized_values = normalized_values / np.max(normalized_values)
    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        if invert_y:
            y = h - y - 1
        if invert_x:
            x = w - x - 1

        xy = np.array([x, y])

        val = normalized_values[state_idx]
        og_val = v_values[state_idx]
        if text_values:
            xy_text = xy
            ax.text(xy_text[0], xy_text[1], '%.1f' % og_val, size='small')
        color = PLOT_CMAP(val)
        ax.add_patch(Rectangle(xy - 0.5, 1, 1, color=color))
    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))
    plt.grid()
    if title:
        plt.title(title)
    plt.show()

